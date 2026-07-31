#include <cmath>
#include <chrono>
#include <condition_variable>
using std::condition_variable;

#include <fstream>

#include <iomanip>
using std::setw;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/ina219.hxx"
#include "common/log.hxx"
#include "rnn/rnn_genome.hxx"
#include "time_series/time_series.hxx"

vector<string> arguments;

vector<vector<vector<double>>> testing_inputs;
vector<vector<vector<double>>> testing_outputs;

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);

    string genome_filename;
    get_argument(arguments, "--genome_file", true, genome_filename);
    RNN_Genome* genome = new RNN_Genome(genome_filename);

    vector<string> testing_filenames;
    get_argument_vector(arguments, "--testing_filenames", true, testing_filenames);

    TimeSeriesSets* time_series_sets = TimeSeriesSets::generate_test(
        testing_filenames, genome->get_input_parameter_names(), genome->get_output_parameter_names()
    );
    Log::debug("got time series sets.\n");

    string normalize_type = genome->get_normalize_type();
    if (normalize_type.compare("min_max") == 0) {
        time_series_sets->normalize_min_max(genome->get_normalize_mins(), genome->get_normalize_maxs());
    } else if (normalize_type.compare("avg_std_dev") == 0) {
        time_series_sets->normalize_avg_std_dev(
            genome->get_normalize_avgs(), genome->get_normalize_std_devs(), genome->get_normalize_mins(),
            genome->get_normalize_maxs()
        );
    }

    Log::info("normalized type: %s \n", normalize_type.c_str());

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    time_series_sets->export_test_series(time_offset, testing_inputs, testing_outputs);

    // Count total rows loaded for testing (across all files)
    int32_t total_rows = 0;
    for (int32_t i = 0; i < (int32_t) testing_inputs.size(); i++) {
        if (testing_inputs[i].size() > 0) {
            total_rows += testing_inputs[i][0].size();
        }
    }
    Log::info("loaded %d total rows across %d test files.\n", total_rows, (int32_t) testing_filenames.size());

    vector<double> best_parameters = genome->get_best_parameters();
    Log::info("Parameter count: %zu\n", best_parameters.size());

    bool use_ina219 = argument_exists(arguments, "--ina219");
    string ina219_device = "/dev/i2c-1";
    get_argument(arguments, "--ina219_device", false, ina219_device);

    INA219 ina219;
    bool ina219_active = false;
    if (use_ina219) {
        if (ina219.open_device(ina219_device.c_str()) && ina219.configure()) {
            ina219_active = true;
            Log::info("INA219 power monitor enabled on %s\n", ina219_device.c_str());
        } else {
            Log::warning(
                "INA219 requested but could not open %s — continuing without power monitoring\n",
                ina219_device.c_str()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Per-file evaluation: MSE, MAE, inference speed, and INA219 power.
    // Each file is evaluated independently so we get file-level granularity.
    // -----------------------------------------------------------------------

    // ----- Idle power baseline measurement ----------------------------------
    // Measure the Pi's quiescent power draw with no inference running.
    // We subtract this from per-file inference energy so the reported
    // energy reflects only the cost of running the model.
    double idle_power_mw = 0.0;
    if (ina219_active) {
        Log::info("Measuring idle power baseline for 2 s (1 ms sampling)...\n");
        INA219Sampler idle_sampler;
        idle_sampler.set_sample_interval_us(1000); // 1 ms — must match inference sampler
        idle_sampler.start(&ina219);
        std::this_thread::sleep_for(std::chrono::seconds(2));
        idle_sampler.stop();
        INA219Stats idle_stats = idle_sampler.get_stats();
        if (idle_stats.sample_count > 0) {
            idle_power_mw = idle_stats.power_mw_avg;
            Log::info(
                "  Idle baseline: %.3f mW  (avg over %d samples)\n",
                idle_power_mw, idle_stats.sample_count
            );
        } else {
            Log::warning("  No idle samples collected — baseline will be 0 mW\n");
        }
    }

    // Accumulators for computing overall averages across all files
    double sum_mse               = 0.0;
    double sum_mae               = 0.0;
    double sum_inference_seconds = 0.0;
    double sum_gross_power_mw    = 0.0;  // total system power during inference
    double sum_net_energy_mj     = 0.0;  // inference-only energy (idle subtracted)
    double sum_gross_energy_mj   = 0.0;  // raw integrated energy
    double sum_current_ma        = 0.0;
    double sum_per_dp_us         = 0.0;
    int    files_with_power      = 0;    // files where INA219 got ≥1 sample
    int    files_with_net_energy = 0;    // files where net_energy >= 0 (valid subtraction)



    // Model latency clock: covers the full loop + write_predictions
    auto model_latency_start = std::chrono::high_resolution_clock::now();

    int32_t n_files = (int32_t) testing_inputs.size();
    for (int32_t fi = 0; fi < n_files; fi++) {
        // Build single-file subvectors so get_mse/get_mae operate on one file
        vector<vector<vector<double>>> file_inputs  = { testing_inputs[fi]  };
        vector<vector<vector<double>>> file_outputs = { testing_outputs[fi] };

        // Row count for this file
        int32_t file_rows = (testing_inputs[fi].size() > 0)
                                ? (int32_t) testing_inputs[fi][0].size()
                                : 0;

        // --- per-file inference timing + INA219 sampling ---
        // Run N_REPS forward passes inside one timed+INA219 window.
        //   Window ≈ N_REPS × single-pass time (~19 ms × 100 ≈ 2 s).
        //   At 1 ms sampling that gives ~2000 power readings — enough for
        //   stable statistics. Divide time and energy by N_REPS afterward
        //   to recover per-inference numbers.
        //   Predictions from the LAST iteration are used for MSE/MAE
        //   (the network is deterministic so every rep gives the same result).
        static const int N_REPS = 100;
        INA219Sampler file_sampler;
        file_sampler.set_sample_interval_us(1000); // 1 ms
        auto inf_start = std::chrono::high_resolution_clock::now();
        if (ina219_active) {
            file_sampler.start(&ina219);
        }

        vector<vector<double>> file_preds;
        for (int rep = 0; rep < N_REPS; rep++) {
            file_preds = genome->get_predictions(best_parameters, file_inputs, file_outputs);
        }

        auto inf_end = std::chrono::high_resolution_clock::now();
        if (ina219_active) {
            file_sampler.stop();
        }

        // --- compute MSE and MAE from predictions (no additional forward pass) ---
        // file_outputs[0] layout: [feature][time_step]
        // file_preds[0]   layout: flat [t*n_outputs + out] interleaved
        double file_mse = 0.0;
        double file_mae = 0.0;
        {
            const vector<double>& preds      = file_preds[0];
            const vector<vector<double>>& exp = file_outputs[0]; // [feature][t]
            int32_t n_outputs = (int32_t) exp.size();
            int32_t n_steps   = (n_outputs > 0) ? (int32_t) exp[0].size() : 0;
            int32_t total     = n_outputs * n_steps;

            if (total > 0 && (int32_t) preds.size() == total) {
                double mse_sum = 0.0, mae_sum = 0.0;
                for (int32_t t = 0; t < n_steps; t++) {
                    for (int32_t o = 0; o < n_outputs; o++) {
                        double diff = preds[t * n_outputs + o] - exp[o][t];
                        mse_sum += diff * diff;
                        mae_sum += std::fabs(diff);
                    }
                }
                file_mse = mse_sum / total;
                file_mae = mae_sum / total;
            } else {
                Log::warning(
                    "MSE/MAE size mismatch for file %d: preds.size()=%zu expected %d "
                    "(n_outputs=%d n_steps=%d). Check prediction layout assumption.\n",
                    fi, preds.size(), total, n_outputs, n_steps
                );
            }
        }

        // Divide total wall-time by N_REPS to get per-inference latency
        auto   inf_duration    = std::chrono::duration_cast<std::chrono::microseconds>(inf_end - inf_start);
        double total_inf_sec   = inf_duration.count() / 1000000.0;
        double file_inf_sec    = total_inf_sec / N_REPS;
        double file_inf_ms     = file_inf_sec * 1000.0;
        double file_dp_us      = (file_rows > 0) ? ((inf_duration.count() / (double) N_REPS) / file_rows) : 0.0;
        double file_dp_ms      = file_dp_us / 1000.0;
        double file_throughput = (file_rows > 0 && file_inf_sec > 0.0) ? (file_rows / file_inf_sec) : 0.0;

        string fname     = testing_filenames[fi];
        size_t slash_pos = fname.find_last_of("/\\");
        string basename  = (slash_pos != string::npos) ? fname.substr(slash_pos + 1) : fname;

        Log::info("------------------------------------------------------------\n");
        Log::info("[File %d/%d] %s\n", fi + 1, n_files, basename.c_str());
        Log::info("  Rows:       %d\n", file_rows);
        Log::info("  MSE:        %.8lf\n", file_mse);
        Log::info("  MAE:        %.8lf\n", file_mae);
        Log::info(
            "  Inference (%d reps, avg):  %.3f s (%.1f ms)  |  per-point: %.3f ms (%.1f us)  |  throughput: %.1f pts/s\n",
            N_REPS, file_inf_sec, file_inf_ms, file_dp_ms, file_dp_us, file_throughput
        );

        // Read CPU frequency to verify the governor stayed pinned during inference.
        // On Linux: /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq (kHz)
        {
            std::ifstream freq_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
            if (freq_file.is_open()) {
                int64_t freq_khz = 0;
                freq_file >> freq_khz;
                Log::info("  CPU freq:   %ld kHz (%.0f MHz)\n", (long) freq_khz, freq_khz / 1000.0);
            } else {
                Log::info("  CPU freq:   (scaling_cur_freq not readable on this OS)\n");
            }
        }


        if (ina219_active) {
            INA219Stats ps = file_sampler.get_stats();
            if (ps.sample_count > 0) {
                // Divide gross energy by N_REPS to get per-inference energy
                double gross_energy_mj = ps.energy_mj / N_REPS;

                // Idle contribution for one inference run
                double idle_energy_mj = idle_power_mw * file_inf_sec; // mW * s = mJ

                // Net energy: cost attributable to running the model
                double net_energy_mj = gross_energy_mj - idle_energy_mj;
                if (net_energy_mj < 0.0) {
                    Log::warning(
                        "  INA219: net energy is negative (%.6f mJ) for %s — "
                        "idle baseline (%.3f mW) may be higher than inference power. "
                        "Re-measure idle with the Pi under equivalent load.\n",
                        net_energy_mj, basename.c_str(), idle_power_mw
                    );
                    // Keep the raw (negative) value in the log so the bug is visible;
                    // clamp only for accumulator so totals don't go nonsensical.
                    sum_gross_power_mw  += ps.power_mw_avg;
                    sum_gross_energy_mj += gross_energy_mj;
                    // do NOT accumulate net energy for this file
                    sum_current_ma      += ps.current_ma_avg;
                    files_with_power++;
                    Log::info(
                        "    Net energy (inference only):  %.6f mJ  (NEGATIVE — see warning above)\n",
                        net_energy_mj
                    );
                    // skip remaining per-file INA219 log lines to avoid confusing output
                } else {
                    Log::info("  INA219 (%d samples @ 1 ms, over %d reps):\n", ps.sample_count, N_REPS);
                    Log::info(
                        "    Voltage: avg %.3f V  (min %.3f, max %.3f)\n",
                        ps.bus_voltage_v_avg, ps.bus_voltage_v_min, ps.bus_voltage_v_max
                    );
                    Log::info(
                        "    Current: avg %.3f mA  (min %.3f, max %.3f)\n",
                        ps.current_ma_avg, ps.current_ma_min, ps.current_ma_max
                    );
                    Log::info(
                        "    Power:   avg %.3f mW  (min %.3f, max %.3f)\n",
                        ps.power_mw_avg, ps.power_mw_min, ps.power_mw_max
                    );
                    Log::info(
                        "    Gross energy (integral P dt):  %.6f mJ\n", gross_energy_mj
                    );
                    Log::info(
                        "    Idle contribution (%.3f mW x %.3f s): %.6f mJ\n",
                        idle_power_mw, file_inf_sec, idle_energy_mj
                    );
                    Log::info(
                        "    Net energy (inference only):  %.6f mJ  |  per-point: %.6f mJ\n",
                        net_energy_mj, (file_rows > 0) ? (net_energy_mj / file_rows) : 0.0
                    );

                    sum_gross_power_mw  += ps.power_mw_avg;
                    sum_gross_energy_mj += gross_energy_mj;
                    sum_net_energy_mj   += net_energy_mj;
                    sum_current_ma      += ps.current_ma_avg;
                    files_with_power++;
                    files_with_net_energy++;
                } // end net_energy >= 0 branch

            } else {
                Log::warning(
                    "  INA219: 0 samples collected for this file "
                    "(inference too short for 1 ms interval?)\n"
                );
            }
        }

        // Accumulate for overall averages
        sum_mse               += file_mse;
        sum_mae               += file_mae;
        sum_inference_seconds += file_inf_sec;
        sum_per_dp_us         += file_dp_us;
    }

    // Write prediction CSVs (once, after all per-file stats are logged)
    genome->write_predictions(
        output_directory, testing_filenames, best_parameters, testing_inputs, testing_outputs, time_series_sets
    );
    auto model_latency_end = std::chrono::high_resolution_clock::now();

    // -----------------------------------------------------------------------
    // Overall averages across all files
    // -----------------------------------------------------------------------
    Log::info("============================================================\n");
    Log::info("OVERALL AVERAGES across %d files:\n", n_files);
    if (n_files > 0) {
        Log::info("  Avg MSE:        %.8lf\n", sum_mse / n_files);
        Log::info("  Avg MAE:        %.8lf\n", sum_mae / n_files);
        Log::info(
            "  Avg inference:  %.3f s  |  avg per-point: %.3f us  |  combined throughput: %.1f pts/s\n",
            sum_inference_seconds / n_files,
            sum_per_dp_us / n_files,
            // Correct aggregate throughput: total data points / total inference time
            // (not the mean of per-file rates, which is mathematically wrong)
            (sum_inference_seconds > 0.0) ? (total_rows / sum_inference_seconds) : 0.0
        );
        Log::info("  Total inference (all files): %.3f s\n", sum_inference_seconds);
        if (ina219_active && files_with_power > 0) {
            Log::info(
                "  Idle baseline:       %.3f mW\n", idle_power_mw
            );
            Log::info(
                "  Avg gross power:     %.3f mW  |  Avg current: %.3f mA\n",
                sum_gross_power_mw / files_with_power,
                sum_current_ma / files_with_power
            );
            Log::info(
                "  Total gross energy:  %.6f mJ  |  avg per file: %.6f mJ\n",
                sum_gross_energy_mj, sum_gross_energy_mj / files_with_power
            );
            if (files_with_net_energy > 0) {
                Log::info(
                    "  Total net energy (idle subtracted):  %.6f mJ  |  avg per file: %.6f mJ  (%d/%d files)\n",
                    sum_net_energy_mj, sum_net_energy_mj / files_with_net_energy,
                    files_with_net_energy, files_with_power
                );
            } else {
                Log::warning(
                    "  Net energy: 0 valid files (all had negative net energy — check idle baseline)\n"
                );
            }
        }
    }
    Log::info("============================================================\n");

    if (ina219_active) {
        ina219.close_device();
    }

    // End-to-end pipeline time: from data-ready to all prediction CSVs written.
    // This is NOT model latency — it includes write_predictions disk I/O.
    auto pipeline_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(model_latency_end - model_latency_start);
    double pipeline_seconds = pipeline_duration.count() / 1000000.0;
    double pipeline_ms      = pipeline_seconds * 1000.0;
    double per_dp_pipeline_ms = (total_rows > 0) ? (pipeline_ms / total_rows) : 0.0;

    Log::info(
        "End-to-end pipeline time (all files + write_predictions): %.3f seconds (%.1f ms)\n",
        pipeline_seconds, pipeline_ms
    );
    Log::info("  Per data point: %.3f ms\n", per_dp_pipeline_ms);

    if (Log::at_level(Log::DEBUG)) {
        int32_t length;
        char* byte_array;

        genome->write_to_array(&byte_array, length);

        Log::debug("WROTE TO BYTE ARRAY WITH LENGTH: %d\n", length);

        RNN_Genome* duplicate_genome = new RNN_Genome(byte_array, length);

        vector<double> best_parameters_2 = duplicate_genome->get_best_parameters();
        Log::debug(
            "duplicate MSE: %lf\n", duplicate_genome->get_mse(best_parameters_2, testing_inputs, testing_outputs)
        );
        Log::debug(
            "duplicate MAE: %lf\n", duplicate_genome->get_mae(best_parameters_2, testing_inputs, testing_outputs)
        );
        duplicate_genome->write_predictions(
            output_directory, testing_filenames, best_parameters_2, testing_inputs, testing_outputs, time_series_sets
        );
    }

    Log::release_id("main");
    return 0;
}

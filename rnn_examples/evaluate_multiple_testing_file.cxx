#include <chrono>
#include <condition_variable>
using std::condition_variable;

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

    // Accumulators for computing overall averages across all files
    double sum_mse               = 0.0;
    double sum_mae               = 0.0;
    double sum_inference_seconds = 0.0;
    double sum_throughput        = 0.0;
    double sum_per_dp_us         = 0.0;
    double sum_power_mw          = 0.0;
    double sum_energy_mj         = 0.0;
    double sum_current_ma        = 0.0;
    int    files_with_power      = 0;

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
        INA219Sampler file_sampler;
        auto inf_start = std::chrono::high_resolution_clock::now();
        if (ina219_active) {
            file_sampler.start(&ina219);
        }

        double file_mse = genome->get_mse(best_parameters, file_inputs, file_outputs);
        double file_mae = genome->get_mae(best_parameters, file_inputs, file_outputs);

        auto inf_end = std::chrono::high_resolution_clock::now();
        if (ina219_active) {
            file_sampler.stop();
        }

        auto   inf_duration    = std::chrono::duration_cast<std::chrono::microseconds>(inf_end - inf_start);
        double file_inf_sec    = inf_duration.count() / 1000000.0;
        double file_inf_ms     = file_inf_sec * 1000.0;
        double file_dp_us      = (file_rows > 0) ? (inf_duration.count() / (double) file_rows) : 0.0;
        double file_dp_ms      = file_dp_us / 1000.0;
        double file_throughput = (file_rows > 0 && file_inf_sec > 0.0) ? (file_rows / file_inf_sec) : 0.0;

        // Extract basename for readable output
        string fname     = testing_filenames[fi];
        size_t slash_pos = fname.find_last_of("/\\");
        string basename  = (slash_pos != string::npos) ? fname.substr(slash_pos + 1) : fname;

        Log::info("------------------------------------------------------------\n");
        Log::info("[File %d/%d] %s\n", fi + 1, n_files, basename.c_str());
        Log::info("  Rows:       %d\n", file_rows);
        Log::info("  MSE:        %.8lf\n", file_mse);
        Log::info("  MAE:        %.8lf\n", file_mae);
        Log::info(
            "  Inference:  %.3f s (%.1f ms)  |  per-point: %.3f ms (%.1f us)  |  throughput: %.1f pts/s\n",
            file_inf_sec, file_inf_ms, file_dp_ms, file_dp_us, file_throughput
        );

        if (ina219_active) {
            INA219Stats ps = file_sampler.get_stats();
            if (ps.sample_count > 0) {
                Log::info("  INA219 (%d samples):\n", ps.sample_count);
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
                    "    Energy:  %.6f mJ  |  per-point: %.6f mJ\n",
                    ps.energy_mj, (file_rows > 0) ? (ps.energy_mj / file_rows) : 0.0
                );
                sum_power_mw   += ps.power_mw_avg;
                sum_energy_mj  += ps.energy_mj;
                sum_current_ma += ps.current_ma_avg;
                files_with_power++;
            }
        }

        // Accumulate for overall averages
        sum_mse               += file_mse;
        sum_mae               += file_mae;
        sum_inference_seconds += file_inf_sec;
        sum_throughput        += file_throughput;
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
            "  Avg inference:  %.3f s  |  avg per-point: %.3f us  |  avg throughput: %.1f pts/s\n",
            sum_inference_seconds / n_files,
            sum_per_dp_us / n_files,
            sum_throughput / n_files
        );
        Log::info("  Total inference (all files): %.3f s\n", sum_inference_seconds);
        if (ina219_active && files_with_power > 0) {
            Log::info(
                "  Avg power:   %.3f mW  |  Avg current: %.3f mA\n",
                sum_power_mw / files_with_power, sum_current_ma / files_with_power
            );
            Log::info("  Total energy (all files): %.6f mJ\n", sum_energy_mj);
            Log::info("  Avg energy per file:      %.6f mJ\n", sum_energy_mj / files_with_power);
        }
    }
    Log::info("============================================================\n");

    if (ina219_active) {
        ina219.close_device();
    }

    // Model Latency (end-to-end: input ready -> all predictions written)
    auto model_latency_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(model_latency_end - model_latency_start);
    double model_latency_seconds = model_latency_duration.count() / 1000000.0;
    double model_latency_ms      = model_latency_seconds * 1000.0;
    double per_data_point_model_latency_ms =
        (total_rows > 0) ? (model_latency_ms / total_rows) : 0.0;

    Log::info(
        "Model latency (entire dataset): %.3f seconds (%.1f ms)\n",
        model_latency_seconds, model_latency_ms
    );
    Log::info("  Per data point: %.3f ms\n", per_data_point_model_latency_ms);

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

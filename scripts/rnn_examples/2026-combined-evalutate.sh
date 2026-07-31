#!/bin/sh

cd build

# INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
# OUTPUT_PARAMETERS="Pitch"
INPUT_PARAMETERS="Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int" 
OUTPUT_PARAMETERS="Main_Flm_Int" 
YEAR=2020
DATA_DIR="../datasets/stock/cohort_${YEAR}_aligned"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: $DATA_DIR missing -- run align_cohort.py + transfer" >&2
    exit 1
fi

# ---- collect the pooled test file list (ALL stocks) ------------------------
TEST_FILES=$(ls "$DATA_DIR"/*_val.csv 2>/dev/null | tr '\n' ' ')
N_TEST=$(echo $TEST_FILES | wc -w | tr -d ' ')

if [ "$N_TEST" -lt 1 ]; then
    echo "ERROR: no *_val.csv files found in $DATA_DIR" >&2
    exit 1
fi

echo "pooled evaluation over $N_TEST stocks from $DATA_DIR"

exp_name="../test_output/2026_stock"
for i in 1 2 3 4 5 6 7 8 9 10
do

    genome_name="../scripts/rnn_examples/2026-stock-combined/cohort_${YEAR}/genome_${i}.bin"
    

    out_dir="../test_output/coal_mpi/evaluation_results_new/$i"
    mkdir -p $out_dir
    echo "Evaluating RNN on coal dataset, results will be saved to: "$out_dir

    ./rnn_examples/evaluate_multiple_testing_file \
    --testing_filenames $TEST_FILES \
    --time_offset 1 \
    --ina219 \
    --genome_file $genome_name \
    --input_parameter_names RET VOL_CHANGE BA_SPREAD ILLIQUIDITY sprtrn TURNOVER \
    --output_parameter_names RET \
    --genome_filename $genome_name \
    --output_directory $out_dir \
    --std_message_level INFO \
    --file_message_level INFO
    # Add --ina219 on Raspberry Pi to log voltage/current/power during inference
done
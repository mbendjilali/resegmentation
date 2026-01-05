#!/bin/bash

# Configuration
NUM_TRIALS=${1:-3}  # Number of trials per configuration (default: 3)
GPUS=(0 1 2)
NUM_GPUS=${#GPUS[@]}

# Define the modes
# Dynamically load modes from scripts/ablation.py
modes_str=$(python -c "import sys, os; sys.path.append(os.path.join(os.getcwd(), 'scripts')); from ablation import ABLATION_CONFIGS; print(' '.join(ABLATION_CONFIGS.keys()))")
IFS=' ' read -r -a modes <<< "$modes_str"

if [ ${#modes[@]} -eq 0 ]; then
    echo "Error: No ablation modes found in scripts/ablation.py or failed to load."
    exit 1
fi

# Create logs directory
mkdir -p logs
ABLATION_DIR="./checkpoints/ablation"
mkdir -p "$ABLATION_DIR"

echo "=========================================="
echo "Ablation Study with Multiple Trials"
echo "=========================================="
echo "Number of trials per configuration: $NUM_TRIALS"
echo "Number of GPUs: $NUM_GPUS"
echo "Configurations: ${modes[@]}"
echo "=========================================="

# Function to count completed trials for a mode
count_completed_trials() {
    local mode=$1
    local count=0
    for i in $(seq 0 $((NUM_TRIALS - 1))); do
        result_file="${ABLATION_DIR}/result_${mode}_trial${i}.json"
        if [ -f "$result_file" ]; then
            count=$((count + 1))
        fi
    done
    echo $count
}

# Function to find next trial number to run for a mode
get_next_trial() {
    local mode=$1
    for i in $(seq 0 $((NUM_TRIALS - 1))); do
        result_file="${ABLATION_DIR}/result_${mode}_trial${i}.json"
        if [ ! -f "$result_file" ]; then
            echo $i
            return
        fi
    done
    echo -1  # All trials completed
}

# Build list of tasks (mode, trial) that need to be run
tasks=()
for mode in "${modes[@]}"; do
    completed=$(count_completed_trials "$mode")
    echo "Mode '$mode': $completed/$NUM_TRIALS trials completed"
    
    for trial in $(seq 0 $((NUM_TRIALS - 1))); do
        result_file="${ABLATION_DIR}/result_${mode}_trial${trial}.json"
        if [ ! -f "$result_file" ]; then
            tasks+=("${mode}:${trial}")
        fi
    done
done

if [ ${#tasks[@]} -eq 0 ]; then
    echo "All trials completed! Aggregating results..."
    python scripts/ablation.py --mode all --aggregate
    exit 0
fi

echo ""
echo "Found ${#tasks[@]} trials to run"
echo ""

# Launch tasks in parallel, assigning to GPUs in round-robin
pids=()
task_idx=0

while [ $task_idx -lt ${#tasks[@]} ]; do
    # Wait for available GPU slot if all GPUs are busy
    while [ ${#pids[@]} -ge $NUM_GPUS ]; do
        # Check which processes are still running
        new_pids=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        pids=("${new_pids[@]}")
        
        if [ ${#pids[@]} -ge $NUM_GPUS ]; then
            sleep 2
        fi
    done
    
    # Get next task
    task="${tasks[$task_idx]}"
    IFS=':' read -r mode trial <<< "$task"
    
    # Assign to GPU
    gpu_idx=$((${#pids[@]} % NUM_GPUS))
    gpu="${GPUS[$gpu_idx]}"
    
    echo "Launching: mode='$mode', trial=$trial on GPU $gpu"
    
    # Run in background
    CUDA_VISIBLE_DEVICES=$gpu python scripts/ablation.py --mode "$mode" --trial "$trial" > "logs/ablation_${mode}_trial${trial}.log" 2>&1 &
    pid=$!
    pids+=("$pid")
    
    task_idx=$((task_idx + 1))
    
    # Small sleep to prevent race conditions
    sleep 2
done

echo ""
echo "All tasks launched. Waiting for completion..."
echo "Monitor progress with: tail -f logs/*.log"

# Wait for all processes to complete
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo ""
echo "All trials completed!"
echo ""

# Aggregate results
echo "Aggregating results..."
python scripts/ablation.py --mode all --aggregate

echo ""
echo "Done! Check ${ABLATION_DIR}/aggregated_stats.json for statistics."

#!/bin/bash
# run_inference_parallel.sh
# Usage: ./run_inference_parallel.sh [arguments for pipeline_inference.py]
# Example: ./run_inference_parallel.sh -i data/test -o data/pred -c checkpoints/best_model.pt -b 64

# Ensure we can trap and clean up background jobs
set -m

# Detect GPUs
if command -v nvidia-smi &> /dev/null; then
    # Get list of GPU indices
    GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader))
else
    echo "nvidia-smi not found. Defaulting to GPU 0."
    GPUS=(0)
fi

NUM_GPUS=${#GPUS[@]}
echo "Found $NUM_GPUS GPUs: ${GPUS[@]}"

# Define workers per GPU (can be adjusted based on CPU cores and memory)
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
TOTAL_SHARDS=$((NUM_GPUS * WORKERS_PER_GPU))

echo "Launching $WORKERS_PER_GPU workers per GPU (Total shards: $TOTAL_SHARDS)"

# Track PIDs and install signal handler to terminate them on Ctrl+C
pids=()

cleanup() {
    echo ""
    echo "Termination requested. Stopping all shards..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Give them a moment to exit gracefully
    sleep 1
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    exit 130
}

trap cleanup INT TERM

# Loop over shards and launch workers

for (( i=0; i<TOTAL_SHARDS; i++ )); do
    # Round-robin assignment to GPUs
    gpu_idx=$((i % NUM_GPUS))
    gpu_id=${GPUS[$gpu_idx]}
    
    echo "Launching shard $i/$TOTAL_SHARDS on GPU $gpu_id..."
    
    # Run the python script in background
    # Pass all script arguments ($@) and append sharding info
    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/pipeline_inference.py "$@" \
        --num_shards "$TOTAL_SHARDS" \
        --shard_id "$i" &
        
    pids+=($!)
done

echo "All shards launched. Waiting for completion..."

# Wait for all background processes
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "Inference complete."


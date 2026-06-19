#!/bin/bash
# SGLang benchmark script - supports generated-shared-prefix, random, random-ids
set -euo pipefail

CONTAINER="${CONTAINER:-sglang-fi-src-128k}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
BASE_URL="http://${HOST}:${PORT}"

# --- Dataset ---
DATASET="${DATASET:-generated-shared-prefix}"   # generated-shared-prefix | random | random-ids

# --- generated-shared-prefix params ---
GSP_NUM_GROUPS="${GSP_NUM_GROUPS:-1}"
GSP_SYSTEM_PROMPT_LEN="${GSP_SYSTEM_PROMPT_LEN:-}"   # unset = use INPUT_LEN
GSP_QUESTION_LEN="${GSP_QUESTION_LEN:-128}"
GSP_NUM_TURNS="${GSP_NUM_TURNS:-1}"
BENCH_BACKEND="${BENCH_BACKEND:-sglang-oai-chat}"    # required for multi-turn

# --- Workload ---
INPUT_LEN="${INPUT_LEN:-131072}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
NUM_PROMPTS="${NUM_PROMPTS:-12}"
CONCURRENCY="${CONCURRENCY:-1}"
RANGE_RATIO="${RANGE_RATIO:-1.0}"
WARMUP="${WARMUP:-1}"

# --- Output ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_IN_CONTAINER="/tmp/bench_${DATASET}_${INPUT_LEN}x${OUTPUT_LEN}_c${CONCURRENCY}_g${GSP_NUM_GROUPS}_${TIMESTAMP}.jsonl"
OUT_HOST="/opt/models/results/bench_${DATASET}_${INPUT_LEN}x${OUTPUT_LEN}_c${CONCURRENCY}_g${GSP_NUM_GROUPS}_${TIMESTAMP}.jsonl"

mkdir -p /opt/models/results

SYSTEM_PROMPT_LEN="${GSP_SYSTEM_PROMPT_LEN:-$INPUT_LEN}"

echo "=============================================="
echo "  SGLang Benchmark"
echo "  Dataset:    $DATASET"
echo "  Backend:    $BENCH_BACKEND"
echo "  Input:      ${INPUT_LEN} (system: ${SYSTEM_PROMPT_LEN}, question: ${GSP_QUESTION_LEN})"
echo "  Output:     $OUTPUT_LEN"
echo "  Prompts:    $NUM_PROMPTS"
echo "  Concurrency: $CONCURRENCY"
echo "  Groups:     $GSP_NUM_GROUPS"
echo "  Turns:      $GSP_NUM_TURNS"
echo "  Range:      $RANGE_RATIO"
echo "  Host:       $BASE_URL"
echo "  Output:     $OUT_HOST"
echo "=============================================="

case "$DATASET" in
    generated-shared-prefix)
        docker exec \
            "$CONTAINER" python3 -m sglang.bench_serving \
            --backend "$BENCH_BACKEND" \
            --dataset-name generated-shared-prefix \
            --gsp-system-prompt-len "$SYSTEM_PROMPT_LEN" \
            --gsp-question-len "$GSP_QUESTION_LEN" \
            --gsp-output-len "$OUTPUT_LEN" \
            --gsp-range-ratio "$RANGE_RATIO" \
            --gsp-num-groups "$GSP_NUM_GROUPS" \
            --gsp-prompts-per-group "$((NUM_PROMPTS / GSP_NUM_GROUPS))" \
            --gsp-num-turns "$GSP_NUM_TURNS" \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$CONCURRENCY" \
            --warmup-requests "$WARMUP" \
            --flush-cache \
            --output-details \
            --output-file "$OUT_IN_CONTAINER"
        ;;
    random)
        docker exec \
            "$CONTAINER" python3 -m sglang.bench_serving \
            --backend "$BENCH_BACKEND" \
            --dataset-name random \
            --dataset-path "${DATASET_PATH:-/tmp/sharegpt.json}" \
            --random-input-len "$INPUT_LEN" \
            --random-output-len "$OUTPUT_LEN" \
            --random-range-ratio "$RANGE_RATIO" \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$CONCURRENCY" \
            --warmup-requests "$WARMUP" \
            --flush-cache \
            --output-details \
            --output-file "$OUT_IN_CONTAINER"
        ;;
    random-ids)
        docker exec \
            "$CONTAINER" python3 -m sglang.bench_serving \
            --backend "$BENCH_BACKEND" \
            --dataset-name random-ids \
            --random-input-len "$INPUT_LEN" \
            --random-output-len "$OUTPUT_LEN" \
            --random-range-ratio "$RANGE_RATIO" \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$CONCURRENCY" \
            --warmup-requests "$WARMUP" \
            --flush-cache \
            --output-details \
            --output-file "$OUT_IN_CONTAINER"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

# Copy result to host
docker cp "$CONTAINER:$OUT_IN_CONTAINER" "$OUT_HOST" 2>/dev/null || true

if [ -f "$OUT_HOST" ]; then
    echo ""
    echo "=== Results ==="
    python3 -c "
import json
with open('$OUT_HOST') as f:
    summary = json.loads(f.readline())
print(f\"Completed:           {summary.get('completed')}/{summary.get('completed',0)}\")
print(f\"Duration:            {summary.get('duration',0):.1f}s\")
print(f\"Input throughput:    {summary.get('input_throughput',0):.0f} tok/s\")
print(f\"Output throughput:   {summary.get('output_throughput',0):.1f} tok/s\")
print(f\"Mean E2E:            {summary.get('mean_e2e_latency_ms',0):.0f} ms\")
print(f\"Median E2E:          {summary.get('median_e2e_latency_ms',0):.0f} ms\")
print(f\"Mean TPOT:           {summary.get('mean_tpot_ms',0):.1f} ms\")
print(f\"Median TPOT:         {summary.get('median_tpot_ms',0):.1f} ms\")
print(f\"P95 TPOT:            {summary.get('p95_tpot_ms',0):.1f} ms\")
print(f\"Max concurrent:      {summary.get('max_concurrent_requests',0)}\")
print(f\"Total input tokens:  {summary.get('total_input_tokens',0):,}\")
print(f\"Total output tokens: {summary.get('total_output_tokens',0):,}\")
"
else
    echo "Failed to copy results from container"
fi

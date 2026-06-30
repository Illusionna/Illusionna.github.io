TARGET_PATH="./models/Qwen3.5-122B-A10B-INT4-AutoRound-Intel"
DRAFT_PATH="./models/Qwen3.5-122B-A10B-DFlash"
TARGET_NAME=$(basename "$TARGET_PATH")
DRAFT_NAME=$(basename "$DRAFT_PATH")
VLLM_IMAGE="i-dgx-spark-gb10:vllm.dflash"

docker run \
    --rm -it \
    --gpus all \
    --name ${TARGET_NAME}-dflash \
    -v $(pwd)/${TARGET_PATH}:/models/${TARGET_NAME} \
    -v $(pwd)/${DRAFT_PATH}:/models/${DRAFT_NAME} \
    -p 4321:4321 \
    --ipc=host \
    --health-cmd 'curl -sf http://localhost:4321/health || exit 1' \
    --health-interval 10s \
    --health-timeout 5s \
    --health-retries 60 \
    --health-start-period 900s \
    --entrypoint vllm ${VLLM_IMAGE} \
        serve /models/${TARGET_NAME} \
        --served-model-name ${TARGET_NAME} \
        --host 0.0.0.0 \
        --port 4321 \
        --gpu-memory-utilization 0.8 \
        --max-model-len 262144 \
        --max-num-seqs 4 \
        --max-num-batched-tokens 16384 \
        --speculative-config '{"model": "/models/'${DRAFT_NAME}'", "method": "dflash", "num_speculative_tokens": 8}' \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --attention-backend FLASH_ATTN \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --enable-auto-tool-choice \
        --trust-remote-code

# HF_HUB_OFFLINE=1 tool-eval-bench \
#     --base-url http://127.0.0.1:4321 \
#     --short --perf \
#     --benchy-args "--tokenizer /home/ai/Illusionna/Desktop/models/Qwen3.5-122B-A10B-INT4-AutoRound-Intel"

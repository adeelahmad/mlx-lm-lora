(
# --- 1. TRAP: RESTORE SYSTEM ON EXIT (Ctrl+C) ---
function cleanup {
    echo -e "\n\033[1;33m🧹 Restoration Sequence Initiated...\033[0m"
    sudo mdutil -i on / > /dev/null 2>&1
    sudo sysctl debug.lowpri_throttle_enabled=1 > /dev/null

    # Resume services
    sudo killall -CONT mediaanalysisd > /dev/null 2>&1
    sudo killall -CONT photoanalysisd > /dev/null 2>&1

    echo "✓ Spotlight & Throttling Re-enabled"
    echo "✓ Background Services Resumed"
    exit
}
trap cleanup SIGINT SIGTERM

# --- 2. KERNEL OVERDRIVE SETTINGS ---
echo -e "\033[1;31m☢️  ENGAGING NUCLEAR PERFORMANCE MODE ☢️\033[0m"

# Force system to treat background tasks as critical
sudo sysctl debug.lowpri_throttle_enabled=0 > /dev/null

# Massive file descriptor limits
sudo sysctl -w kern.maxfiles=9991520 > /dev/null
sudo sysctl -w kern.maxfilesperproc=995760 > /dev/null

# GPU Wired Memory Override (Only for >96GB RAM)
sudo sysctl iogpu.wired_limit_mb=80536 > /dev/null 2>&1

# Network & IO Tuning
sudo sysctl -w net.inet.tcp.delayed_ack=0 > /dev/null
sudo sysctl -w kern.ipc.somaxconn=2048 > /dev/null

# --- 3. KILL BACKGROUND NOISE ---
# Disable Spotlight
sudo mdutil -i off / > /dev/null 2>&1

# Pause heavy Apple Intelligence/Media daemons (Errors suppressed if not running)
sudo killall -STOP mediaanalysisd > /dev/null 2>&1 || true
sudo killall -STOP photoanalysisd > /dev/null 2>&1 || true
sudo killall -STOP distnoted > /dev/null 2>&1 || true
sudo tmutil stopbackup > /dev/null 2>&1 || true

# --- 4. THE INFINITE LOOP ---
echo -e "\033[1;32m🤖 SYSTEM LOCKED. STARTING TRAINING.\033[0m"

# Wrap in caffeinate to prevent sleep/idle
caffeinate -i -s -m bash -c '
while true; do
    # 1. Aggressive RAM Purge before every run
    echo "🧹 Clearing Wired/Inactive Memory..."
    sync && sudo /usr/sbin/purge

    # 2. THE COMMAND
    # Removed "taskpolicy" to fix the crash.
    # "nice -n -20" is sufficient to force P-Core usage.
    # sudo cp /Users/adeelahmad/work/mlx-grpo-trainer/outputsz/checkpoints/best_model/latest/model.safetensors  /Users/adeelahmad/.cache/lm-studio/models/lmstudio-community/Qwen-4B-Thinking-2507.z/ || echo "NO MODEL" && sudo rm -rf /Users/adeelahmad/work/mlx-grpo-trainer/outputsz/checkpoints
    # sudo
    Model=${Model:-/Users/adeelahmad/work/SiLLM-examples/helpsteer/mlx-grpo/outy1266_align_last32/latest}
    sudo nice -n -20 env \
        OMP_NUM_THREADS=16 \
        OMP_THREAD_LIMIT=16 \
        MKL_NUM_THREADS=16 \
        NUMEXPR_NUM_THREADS=16 \
        TOKENIZERS_PARALLELISM=false \
        PYTORCH_ENABLE_MPS_FALLBACK=1 \
        mlx_lm_lora.train \
                  --model $Model \
                  --reference-model-path $Model \
                   --train \
                  --data /Users/adeelahmad/work/SiLLM-examples/helpsteer/mlx-grpo/strat \
                   --train-mode grpo \
                  --grpo-loss-type dr_grpo \
                  --group-size 2 \
                  --epsilon 1e-4 \
                  --epsilon-high 0.05 --beta 0.03 \
                  --temperature 0.8 \
                  --learning-rate 1e-6 \
                  --max-seq-length 1024 \
                  --importance-sampling-level sequence \
                  --gradient-accumulation-steps 1 \
                  --steps-per-report 1 \
                  --steps-per-eval 50 \
                  --wandb mlx-lm-grpo-v4.9 \
                  --save-every 4 \
                  --iters 1000 \
                  --batch-size 1 \
                  --seed $RANDOM \
                  --val-batches 1 \
                  --fuse \
                  --adapter-path adapters/turn4 \
                  --optimizer adamw \
                  --num-layers -1  \
                  --reward-functions "r1_semantic_similarity_reward,r1_conditional_content_reward,r1_velocity_to_correct_thinking_reward,r1_format_reward,r1_tag_structure_reward,r1_thinking_quality_reward" \
                  --reward-weights "[0.25, 0.25, 0.20, 0.10, 0.10, 0.10]" --train-type lora --load-in-4bits \
                  --num-actors 2 --actor-quantizations "2bit,2bit" --actor-kl-to-main-weight 0.1 --actor-sync-mode main_to_actors


        # sudo cp /Users/adeelahmad/work/mlx-grpo-trainer/outputsz/checkpoints/best_model/latest/model.safetensors  /Users/adeelahmad/.cache/lm-studio/models/lmstudio-community/Qwen-4B-Thinking-2507.z/ || echo "NO MODEL" && sudo rm -rf /Users/adeelahmad/work/mlx-grpo-trainer/outputsz/checkpoints

    # 3. Stop Logic
    if [ -f stop_trainer ]; then
        rm stop_trainer
        echo "🛑 Stop file detected. Exiting loop..."
        break
    fi

    echo "Sleeping for 15 sec..."
    sleep 15
done
'

# Run cleanup if the loop breaks naturally
cleanup
)

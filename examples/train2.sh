export WANDB_API_KEY="dd71f286a1b06ec9081aa8ac585c09735f785fcd"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1 # (可选) 禁用 P2P 也就是 NVLink，有时候能绕过硬件 bug
export CUDA_VISIBLE_DEVICES=0,1,2,3

mp_size=1   # Must be 1 for now
dp_size=2
pp_size=2

exp_name=test-opensae-trainer-tensor-off

hookpoint=layers.26
exit_layer=26
base_model="/home/hujw/Qwen3-1.7B"           # layer nums: 0 - 31
dataset="/data/hujw/training.mmap"


MODEL_CONFIG="--model ${base_model} \
--auto_model_class AutoModel \
--model.trust_remote_code True
"

DATA_CONFIG="--dataset ${dataset} \
--split train \
--ctx_len 4096 \
--data.trust_remote_code True
"

# fastest configuration: local batch size = 4, micro acc steps = 4 
TRAIN_CONFIG="
--hookpoint ${hookpoint} \
--mp_size ${mp_size} \
--dp_size ${dp_size} \
--pp_size ${pp_size} \
--fsdp False \
--adam_in_8bit False \
--local_batch_size 8 \
--global_batch_size 256 \
--micro_acc_steps 4 \
--distribute_modules True \
--save_every 500 \
--save_dir /data/hujw/CHECKPOINTS9 \
--load_dir /data/hujw/CHECKPOINTS9 \
--dead_feature_threshold 10000000 \
--multi_topk True \
--k_scheduler constant \
--k_scheduler_step_ratio 0.1 \
--auxk_alpha 1e-2 \
--lr_scheduler_type wsd \
--lr_warmup_ratio 0.1 \
--lr_decay_ratio 0.05 \
--spike_detection_start 200000 \
--spike_detection_window_size 5 \
--spike_detection_threshold_ratio 1.8 \
--varlen True \
--early_exit_inference_layer_num ${exit_layer} \
--log_to_wandb True \
--wandb_project SAE_FOR_QWEN3_9025 \
--wandb_log_frequency 1 \
"



### For OpenAI
### Expansion Factor is near 170 for GPT-2, thus, n = 131072.
### Sparsity k is 128.

### For eleuther AI, they use expansion_factor = 32, k = 128.
SAE_CONFIG="--expansion_factor 48
--k 128
--normalize True
--shift_back False
"







export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"



torchrun_arguments="\
    --nproc_per_node $((mp_size * dp_size*pp_size)) \
    --master-port 10045 \
    -m opensae.trainer \
        ${MODEL_CONFIG} ${DATA_CONFIG} \
        ${TRAIN_CONFIG} ${SAE_CONFIG} \
        --run_name $exp_name
    "
echo $torchrun_arguments


torchrun $torchrun_arguments
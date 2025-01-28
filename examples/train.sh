export WANDB_API_KEY="YOUR WANDB API KEY"

mp_size=1   # Must be 1 for now
dp_size=`ls -al /dev/nvidia* | grep nvidia[0-9] | wc -l`

exp_name=test-opensae-trainer

hookpoint=layers.2
exit_layer=2
base_model="/MODELS/Meta-Llama-3.1-8B/"           # layer nums: 0 - 31
dataset="/DATA/1b.mmap"


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
--fsdp False \
--adam_in_8bit False \
--local_batch_size 4 \
--global_batch_size 256 \
--micro_acc_steps 4 \
--distribute_modules True \
--save_every 200 \
--save_dir /CHECKPOINTS \
--load_dir /CHECKPOINTS \
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
--log_to_wandb False \
--wandb_project SAE \
"



### For OpenAI
### Expansion Factor is near 170 for GPT-2, thus, n = 131072.
### Sparsity k is 128.

### For eleuther AI, they use expansion_factor = 32, k = 128.
SAE_CONFIG="--expansion_factor 64
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
    --nproc_per_node $((mp_size * dp_size)) \
    --master-port $((10000 + $RANDOM % 100)) \
    -m opensae.trainer \
        ${MODEL_CONFIG} ${DATA_CONFIG} \
        ${TRAIN_CONFIG} ${SAE_CONFIG} \
        --run_name $exp_name
    "
echo $torchrun_arguments


/root/miniconda3/bin/torchrun $torchrun_arguments


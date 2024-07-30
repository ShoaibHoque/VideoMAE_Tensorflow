# Set the path to save checkpoints
OUTPUT_DIR='YOUR_PATH/k400_videomae_pretrain_large_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600'
# Set the path to Kinetics train set. 
DATA_PATH='YOUR_PATH/list_kinetics-400/train.csv'

# Batch size can be adjusted according to the number of GPUs
# This script is for 64 GPUs (8 nodes x 8 GPUs)
OMP_NUM_THREADS=1

# Define the distributed training parameters for TensorFlow
NUM_NODES=8
NUM_GPUS_PER_NODE=8
MASTER_ADDR=$2
MASTER_PORT=12320
NODE_RANK=$1

python run_mae_pretraining.py \
    --data_path ${DATA_PATH} \
    --mask_type tube \
    --mask_ratio 0.9 \
    --model pretrain_videomae_large_patch16_224 \
    --decoder_depth 12 \
    --batch_size 8 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 40 \
    --save_ckpt_freq 20 \
    --epochs 1601 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --num_nodes ${NUM_NODES} \
    --num_gpus_per_node ${NUM_GPUS_PER_NODE} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --node_rank ${NODE_RANK}

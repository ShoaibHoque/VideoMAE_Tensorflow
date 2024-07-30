# Set the path to save checkpoints
OUTPUT_DIR='YOUR_PATH/ssv2_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e800/eval_lr_5e-4_epoch_50'
# Set the path to SSV2 set (train.csv/val.csv/test.csv)
DATA_PATH='YOUR_PATH/list_ssv2'
# Set the path to pretrain model
MODEL_PATH='YOUR_PATH/ssv2_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e800/checkpoint-799.pth'

# Batch size can be adjusted according to the number of GPUs
# This script is for 64 GPUs (8 nodes x 8 GPUs)
NUM_NODES=8
NUM_GPUS_PER_NODE=8
MASTER_ADDR=$2
MASTER_PORT=12320
NODE_RANK=$1

python run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set SSV2 \
    --nb_classes 174 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --num_nodes ${NUM_NODES} \
    --num_gpus_per_node ${NUM_GPUS_PER_NODE} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --node_rank ${NODE_RANK} \
    --enable_deepspeed

# Set the path to save checkpoints
OUTPUT_DIR='YOUR_PATH/k400_videomae_pretrain_huge_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/eval_lr_1e-3_epoch_50_droppath_0.2_layer_decay_0.75'
# Path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='YOUR_PATH/list_kinetics-400'
# Path to pretrain model
MODEL_PATH='YOUR_PATH/k400_videomae_pretrain_huge_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/checkpoint-1599.pth'

# Batch size can be adjusted according to the number of GPUs
# This script is for 64 GPUs (8 nodes x 8 GPUs)
OMP_NUM_THREADS=1

# Define the distributed training parameters for TensorFlow
NUM_NODES=8
NUM_GPUS_PER_NODE=8
MASTER_ADDR=$2
MASTER_PORT=12320
NODE_RANK=$1

python run_class_finetuning.py \
    --model vit_huge_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 400 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 1e-3 \
    --warmup_lr 1e-8 \
    --min_lr 1e-5 \
    --layer_decay 0.75 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --drop_path 0.2 \
    --fc_drop_rate 0.5 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --dist_eval \
    --use_checkpoint \
    --num_nodes ${NUM_NODES} \
    --num_gpus_per_node ${NUM_GPUS_PER_NODE} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --node_rank ${NODE_RANK}

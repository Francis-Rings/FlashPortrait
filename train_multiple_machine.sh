export MODEL_NAME="/path/FlashPortrait/checkpoints/Wan2.1-I2V-14B-720P"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NODE_RANK=${RANK}
echo $NODE_RANK $MASTER_ADDR $MASTER_PORT
NCCL_DEBUG=INFO

accelerate launch --num_machines=12 --num_processes=96 --machine_rank=$RANK --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config_cpu_offload.json --deepspeed_multinode_launcher standard train_portrait.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pd_fpg_model_path="/path/FlashPortrait/checkpoints/FlashPortrait/pd_fpg.pth" \
  --alignment_model_path="/path/FlashPortrait/checkpoints/FlashPortrait/face_landmark.onnx"  \
  --det_model_path="/path/FlashPortrait/checkpoints/FlashPortrait/face_det.onnx" \
  --train_data_square_dir="/path/FlashPortrait/portrait_data/video_square_path.txt" \
  --train_data_rec_dir="/path/FlashPortrait/portrait_data/video_rec_path.txt" \
  --train_data_vec_dir="/path/FlashPortrait/portrait_data/video_vec_path.txt" \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=20 \
  --checkpointing_steps=2000 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_14B_dir" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --uniform_sampling \
  --low_vram \
  --use_deepspeed \
  --train_mode="i2v" \
  --motion_sub_loss \
  --trainable_modules "."

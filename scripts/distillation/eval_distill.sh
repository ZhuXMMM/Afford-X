python -m torch.distributed.launch --master_port=23456 --nproc_per_node=4 --use_env main.py \
    --dataset_config configs/[tdod.json/tdod_coco.json/tdod_lvis.json] \
    --valid_batch_size 12 \
    --load [trained_distill_mask_checkpoint] \
    --ema \
    --num_workers 5 \
    --output-dir 'logs/distill_mask/test' \
    --distillation \
    --cluster \
    --cluster_memory_size 1024 \
    --cluster_num 3 \
    --cluster_feature_loss 1e4 \
    --fusion --verb_att \
    --eval \
    # --load_full \     # For LVIS-Aff
    # --load_word_full \ # For LVIS-Aff&COCO-Aff
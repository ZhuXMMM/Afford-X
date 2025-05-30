python -m torch.distributed.launch --master_port=23456 --nproc_per_node=4 --use_env main.py \
    --dataset_config configs/tdod_coco.json \
    --train_batch_size 4  \
    --valid_batch_size 8 \
    --frozen_weights [trained_distill_checkpoint] \
    --mask_model smallconv \
    --no_aux_loss \
    --ema --text_encoder_lr 1e-8 --lr 5e-5 --lr_fusion 1e-5\
    --num_workers 4 \
    --output-dir 'logs/distill_mask/lvis' \
    --eval_skip 1 \
    --fusion --verb_att \
    --cluster \
    --cluster_memory_size 1024 \
    --cluster_num 3 \
    --no_contrastive_align_loss \
    --load_full  --load_word_full \


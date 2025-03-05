python -m torch.distributed.launch --master_port=23456 --nproc_per_node=4 --use_env main.py \
    --dataset_config configs/tdod_coco.json \
    --train_batch_size 6  \
    --valid_batch_size 12 \
    --frozen_weights [trained_pronoun_checkpoint] \
    --ema --text_encoder_lr 1e-5 --lr 5e-6 --lr_fusion 1e-5\
    --num_workers 4 \
    --output-dir 'logs/pronoun_mask/coco' \
    --eval_skip 1 \
    --mask_model smallconv \
    --no_aux_loss \
    --no_contrastive_align_loss \
    --fusion --verb_att \
    --load_word_full \



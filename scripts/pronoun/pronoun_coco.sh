python -m torch.distributed.launch --master_port=23456 --nproc_per_node=4 --use_env main.py \
    --dataset_config configs/tdod_coco.json \
    --train_batch_size 18  \
    --valid_batch_size 18 \
    --ema --text_encoder_lr 1e-5 --lr 5e-5 --lr_fusion 8e-5\
    --num_workers 4 \
    --output-dir 'logs/sth/coco' \
    --eval_skip 1 \
    --fusion --verb_att \
    --load_word_full \

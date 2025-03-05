python -m torch.distributed.launch --master_port=23456 --nproc_per_node=4 --use_env main.py \
    --dataset_config configs/tdod.json \
    --train_batch_size 8  \
    --valid_batch_size 8 \
    --ema --text_encoder_lr 1e-5 --lr 5e-5 --lr_fusion 8e-5\
    --num_workers 4 \
    --output-dir 'logs/pronoun/cocotasks' \
    --eval_skip 2 \
    --fusion --verb_att \
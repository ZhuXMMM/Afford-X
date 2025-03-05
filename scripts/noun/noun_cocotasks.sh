python -m torch.distributed.launch --master_port=23456 --nproc_per_node=4 --use_env main.py \
    --dataset_config configs/tdod.json \
    --train_batch_size 12  \
    --valid_batch_size 18 \
    --ema --text_encoder_lr 1e-5 --lr 1e-6 --lr_fusion 1e-6 \
    --num_workers 4 \
    --output-dir 'logs/noun/cocotasks' \
    --eval_skip 1 \
    --verb_noun_input \
    --fusion --verb_att \
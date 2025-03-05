python -m torch.distributed.launch --master_port=23456 --nproc_per_node=4 --use_env main.py \
    --dataset_config configs/[tdod.json/tdod_coco.json/tdod_lvis.json] \
    --valid_batch_size 6  \
    --load [trained_noun_checkpoint] \
    --ema \
    --num_workers 5 \
    --output-dir 'logs/noun/test' \
    --no_contrastive_align_loss \
    --verb_noun_input \
    --fusion --verb_att \
    --eval \
    # --load_full \     # For LVIS-Aff
    # --load_word_full \ # For LVIS-Aff&COCO-Aff
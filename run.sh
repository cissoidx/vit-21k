python -u -m torch.distributed.launch --nproc_per_node=8 train_single_label_from_scratch.py \
    --data_path=/home/ubuntu/data/imagenet1k \
    --savename=ckpt-1


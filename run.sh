python -u -m torch.distributed.launch --nproc_per_node=8 train_semantic_softmax.py \
    --batch_size=64 \
    --num_workers=16 \
    --data_path=/home/ubuntu/data/imagenet10k/imagenet21k_resized_new \
    --model_name=vit_base_patch16_224 \
    --num_classes=10450 \
    --epochs=300

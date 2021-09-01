CUDA_VISIBLE_DEVICES=0,1 python train_semantic_softmax.py \
    --batch_size=8 \
    --data_path=/home/ubuntu/data/test21k \
    --model_name=vit_base_patch16_224 \
    --num_classes=19167 \
    --epochs=300

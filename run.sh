dt=$(date '+%d-%m-%Y_%H-%M-%S')
script log_$dt.txt

CUDA_VISIBLE_DEVICES=0,1 python train_semantic_softmax.py \
--batch_size=8 \
--data_path=/home/ubuntu/data/imagenet10k/imagenet21k_resized_new \
--model_name=vit_base_patch16_224 \
--num_classes=10450 \
--epochs=300

exit
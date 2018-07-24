DATASET_DIR=/home/shared/data_zoo/imagenet/tf_records/
CHECKPOINT_FILE=/home/shared/py3_workspace/senet/train_logs/s2
EVAL_DIR=/home/shared/py3_workspace/senet/eval_logs/s2
CUDA_VISIBLE_DEVICES=1 python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --eval_dir=${EVAL_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=resnet_v1_50 \
    --batch_size=128
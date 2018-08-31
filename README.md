# SENet-TensorFlow-Slim
This is a Tensorflow implementation of ["Squeeze-and-Excitation Networks"](https://arxiv.org/pdf/1709.01507) aiming to be compatible on the [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim).

## Squeeze-and-Excitation Networks
**SENet** proposes an architectural unit called *"Squeeze-and-Excitation" (SE)* block to improve the representational power of a network by explicitly modelling the **interdependencies between the channels** of its convolutional features.

### Diagram of a SE-Block
<div align="center">
  <img src="https://github.com/kobiso/SENet-tensorflow-slim/blob/master/figures/senet.jpg">
</div>

### Schema of SE-Inception and SE-ResNet modules
<div align="center">
   <img src="https://github.com/kobiso/SENet-tensorflow-slim/blob/master/figures/inception.png" width="420">
  <img src="https://github.com/kobiso/SENet-tensorflow-slim/blob/master/figures/res.png"  width="420">
</div>

### Single-crop error rates (%) on the ImageNet validation set

<div align="center">
  <img src="https://github.com/kobiso/SENet-tensorflow-slim/blob/master/figures/ex1.png">
</div>

<div align="center">
  <img src="https://github.com/kobiso/SENet-tensorflow-slim/blob/master/figures/ex2.png"  width="420">
</div>

## Prerequisites
- Python 3.x
- TensorFlow 1.x
- TF-slim
  - Check the ['installation' part of TF-Slim image models README](https://github.com/tensorflow/models/tree/master/research/slim#installation).

## Prepare Data set
You should prepare your own dataset or open dataset (Cifar10, flowers, MNIST, ImageNet).
For preparing dataset, you can follow the ['preparing the datasets' part in TF-Slim image models README](https://github.com/tensorflow/models/tree/master/research/slim#preparing-the-datasets).

## SE-block Supportive Models
This project is based on TensorFlow-Slim image classification model library.
Every image classification model in TensorFlow-Slim can be run the same.
And, you can run SE-block added models in the below list by adding one argument `--attention_module='se_block'` when you train or evaluate a model.

- Inception V4 + SE
- Inception-ResNet-v2 + SE
- ResNet V1 50 + SE
- ResNet V1 101 + SE
- ResNet V1 152 + SE
- ResNet V1 200 + SE
- ResNet V2 50 + SE
- ResNet V2 101 + SE
- ResNet V2 152 + SE
- ResNet V2 200 + SE

### Change *Reduction ratio*
To change *reduction ratio*, you have to manually set the ratio on `def se_block(residual, name, ratio=8)` method in `SENet-tensorflow-slim/nets/attention_module.py`.

#### Single-crop error rates (%) on ImageNet validation set at different reduction ratios
<div align="center">
  <img src="https://github.com/kobiso/SENet-tensorflow-slim/blob/master/figures/ex5.png"  width="420">
</div>

## Train a Model
### Train a model with SE-block
Below script gives you an example of training a model with SE-block.
Don't forget to put an argument `--attention_module=se_block`.
```
DATASET_DIR=/DIRECTORY/TO/DATASET
TRAIN_DIR=/DIRECTORY/TO/TRAIN
CUDA_VISIBLE_DEVICES=0 python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=resnet_v1_50 \
    --batch_size=100 \
    --attention_module=se_block
```

### Train a model without SE-block
Below script gives you an example of training a model without SE-block.
```
DATASET_DIR=/DIRECTORY/TO/DATASET
TRAIN_DIR=/DIRECTORY/TO/TRAIN
CUDA_VISIBLE_DEVICES=0 python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=resnet_v1_50 \
    --batch_size=100
```

## Evaluate a Model
To keep track of validation accuracy while training, you can use `eval_image_classifier_loop.py` which evaluate the performance at multiple checkpoints during training.
If you want to just evaluate a model once, you can use `eval_image_classifier.py`.

### Evaluate a model with SE-block
Below script gives you an example of evaluating a model with SE-block during training.
Don't forget to put an argument `--attention_module=se_block`.
```
DATASET_DIR=/DIRECTORY/TO/DATASET
CHECKPOINT_FILE=/DIRECTORY/TO/CHECKPOINT
EVAL_DIR=/DIRECTORY/TO/EVAL
CUDA_VISIBLE_DEVICES=0 python eval_image_classifier_loop.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --eval_dir=${EVAL_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=resnet_v1_50 \
    --batch_size=100 \
    --attention_module=se_block
```

### Evaluate a model without SE-block
Below script gives you an example of evaluating a model without SE-block during training.
```
DATASET_DIR=/DIRECTORY/TO/DATASET
CHECKPOINT_FILE=/DIRECTORY/TO/CHECKPOINT
EVAL_DIR=/DIRECTORY/TO/EVAL
CUDA_VISIBLE_DEVICES=0 python eval_image_classifier_loop.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --eval_dir=${EVAL_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=resnet_v1_50 \
    --batch_size=100 
```

## Related Works
- Blog: [SENet](https://kobiso.github.io//research/research-SENet/)
- Repository: [CBAM-TensorFlow-Slim](https://github.com/kobiso/CBAM-tensorflow-slim)
- Repository: [CBAM-TensorFlow](https://github.com/kobiso/CBAM-tensorflow)
- Repository: [CBAM-Keras](https://github.com/kobiso/CBAM-keras)

## Reference
- Paper: [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507)
- Repository: [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim)
  
## Author
Byung Soo Ko / kobiso62@gmail.com

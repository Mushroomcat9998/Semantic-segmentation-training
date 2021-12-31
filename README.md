# Semantic segmentation training scripts
# Model: deeplabv3_mobilenet_v3_large

## How to use colab file for training:

#### 1 Check GPU specs and mount to the drive that you have zip file of dataset (Cell 1 and 2)

#### 2 Unzip dataset and rename folder contains dataset (Cell 3)
a Dataset should be in COCO format  
b Unzip dataset to the path that correspond with the path in file json in dataset  
c The name that be changed should correspond with the name in file json in dataset

#### 3 Clone training code from remote repo (Cell 4)

#### 4 Start training model (Cell 5)

```
!python train.py \
--data-path /path/to/data/ \
-b batch_size \
-j num_workers \
--epochs num_epochs \
--print-freq num_iters_for_log \
--resume '/path/to/checkpoint/' \
--output-dir /path/for/output/result/ \
--model model_name (default: deeplabv3_mobilenet_v3_large) \
--aux-loss \
--wd weight_decay_coef \
```


# Semantic segmentation reference training scripts

This folder contains reference training scripts for semantic segmentation.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

All models have been trained on 8x V100 GPUs.

You must modify the following flags:

`--data-path=/path/to/dataset`

`--nproc_per_node=<number_of_gpus_available>`

## fcn_resnet50
```
torchrun --nproc_per_node=8 train.py --lr 0.02 --dataset coco -b 4 --model fcn_resnet50 --aux-loss
```

## fcn_resnet101
```
torchrun --nproc_per_node=8 train.py --lr 0.02 --dataset coco -b 4 --model fcn_resnet101 --aux-loss
```

## deeplabv3_resnet50
```
torchrun --nproc_per_node=8 train.py --lr 0.02 --dataset coco -b 4 --model deeplabv3_resnet50 --aux-loss
```

## deeplabv3_resnet101
```
torchrun --nproc_per_node=8 train.py --lr 0.02 --dataset coco -b 4 --model deeplabv3_resnet101 --aux-loss
```

## lraspp_mobilenet_v3_large
```
torchrun --nproc_per_node=8 train.py --dataset coco -b 4 --model lraspp_mobilenet_v3_large --wd 0.000001
```

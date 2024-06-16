
# Adversaries With Incentives: A Strategic Alternative to Adversarial Robustness
<p align="center">
  <img src='https://github.com/7yt8/AWI/blob/main/4d166eeb_1.png'>
</p>

## Requirements

To install requirements:

```setup
conda create --name <env> --file requirements.txt
```


## Training

Following are example commands for trainin Resnet18 on CIFAR-10 using accelerate.

Training a clean model:

```train clean
NET=resnet # resnet | vgg1_bn | ViT
DS=cifar10 # cifar10 | GTSRB
NUM_GPUS=2
STRAT=0 # 0 - clean | 1 - adversarial | 3 - known 1-hot attack | 4 - multi-targeted attacks

accelerate launch --num_processes ${NUM_GPUS} --main_process_port 12345 train.py --training_strategy=${STRAT} --epochs=50 --use_aug=1 --arch=${NET} --dataset=${DS} --num_gpus=${NUM_GPUS} --output_dir=./resnet_models/clean --bs=128
```

Training an Adversarial model:

```train adv
NET=resnet # resnet | vgg1_bn | ViT
DS=cifar10 # cifar10 | GTSRB
NUM_GPUS=2 #
STRAT=1 # 0 - clean | 1 - adversarial | 3 - known 1-hot attack | 4 - multi-targeted attacks
ATTCK=0 # 0 | 8 (for multi-targeted)

accelerate launch --num_processes ${NUM_GPUS} --main_process_port 12345 train.py --training_strategy=${STRAT} --epochs=50 --use_aug=1 --arch=${NET} --dataset=${DS} --num_gpus=${NUM_GPUS} --attack_loss=${ATTCK} --output_dir=./resnet_models/adv --bs=128
```

Training a strategic model against a known 1-hot attack with uncertainty 0.1:

```train adv
NET=resnet # resnet | vgg1_bn | ViT
DS=cifar10 # cifar10 | GTSRB
NUM_GPUS=2 #
STRAT=3 # 0 - clean | 1 - adversarial | 3 - known 1-hot attack | 4 - multi-targeted attacks
GT_ATTACK=1235670894 # the known 1-hot attacks given as a sequence of targets (e.g. label 0 attacks towards label 1. label 9 attacks towards label 4
FP=0.1 # uncertainty

accelerate launch --num_processes ${NUM_GPUS} --main_process_port 12345 train.py --training_strategy=${STRAT} --epochs=50 --use_aug=1 --arch=${NET} --dataset=${DS} --num_gpus=${NUM_GPUS} --output_dir=./resnet_models/known_1hot_${GT_ATTACK} --flip_prob=${FP} --gt_strat 1 2 3 5 6 7 0 8 9 4
```

Training a strategic model against a multi-targeted semantic attacker:

```train multi-targeted
NET=resnet # resnet | vgg1_bn | ViT
DS=cifar10 # cifar10 | GTSRB
NUM_GPUS=1 #
STRAT=4 # 0 - clean | 1 - adversarial | 3 - known 1-hot attack | 4 - multi-targeted attacks
FP=0.1
ATTCK=8
TARGETS_MAT_PATH=./targets_mats/${DS}/targets_mat_semantic # path to a boolean 2d tensor (pickled) describing the multi-targeted attack

accelerate launch --num_processes ${NUM_GPUS} --main_process_port 12345 train.py --training_strategy=${STRAT} --epochs=50 --use_aug=1 --arch=${NET} --dataset=${DS} --num_gpus=${NUM_GPUS} --attack_loss=${ATTCK} --output_dir=./resnet_models/semantic --flip_prob=${FP} --targets_mat_path=${TARGETS_MAT_PATH} --bs=128
```


## Evaluation

To evaluate one a model against clean, adversarial and multi-targeted attacks, run:

```eval
NET=resnet # resnet | vgg1_bn | ViT
DS=cifar10 # cifar10 | GTSRB
MODEL_PATH=./resnet_models/semantic/my_model.pt
TARGETS_MAT_PATH=./targets_mats/${DS}/targets_mat_semantic # path to a boolean 2d tensor (pickled) describing the multi-targeted attack

python eval.py --dataset=${DS} --arch=${NET} --model_path=${MODEL_PATH} --results_dir=./evaluations/ --targets_mat_path=${TARGETS_MAT_PATH} --mt_loss_list 8 --test_loss_list 0
```


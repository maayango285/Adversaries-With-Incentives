

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
GTSRB_MEAN = (0.3402, 0.3121, 0.3214)
GTSRB_STD = (0.2755, 0.2646, 0.2712)
# stats for cifar20 resized to 224x224
CIFAR10_224_MEAN = CIFAR10_MEAN
CIFAR10_224_STD =(0.2413, 0.2378, 0.2564) 

STD_MEAN_DICT = {'cifar10': {'mean': CIFAR10_MEAN, 'std': CIFAR10_STD}, \
                  'GTSRB': {'mean': GTSRB_MEAN, 'std': GTSRB_STD}, \
                  'GTSRB_2g': {'mean': GTSRB_MEAN, 'std': GTSRB_STD}, \
                  'cifar100': {'mean': CIFAR100_MEAN, 'std': CIFAR100_STD}, \
                  'cifar10_224': {'mean': CIFAR10_224_MEAN, 'std': CIFAR10_224_STD}}
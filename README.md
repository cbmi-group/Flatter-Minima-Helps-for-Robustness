# Flatter Minima of Loss Landscapes Correspond with Strong Corruption Robustness
  This repos provides a method to imporve corruption robustness.
## Dependencies
  - python 3
  - torch>=1.4.0
  - torchvision>=0.5.0
## How to use
### 1. Train

python cifar_demo.py  --model UNet  --args.dataset cifar10(cifar100)

### 2. Test

python test.py  --model UNet  --args.dataset cifar10(cifar100)

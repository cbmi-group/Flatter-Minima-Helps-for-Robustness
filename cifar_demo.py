# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Main script to launch PixMix training on CIFAR-10/100.

Supports WideResNet, ResNeXt models on CIFAR-10 and CIFAR-100 as well
as evaluation on CIFAR-10-C and CIFAR-100-C.

Example usage:
  `python cifar.py`
"""
from __future__ import print_function

import argparse
import os
import shutil
import time
import logging

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset',type=str,default='cifar100',choices=['cifar10', 'cifar100'],help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--opti', default='SGD', type=str)
parser.add_argument('--gpus', default='0,1,2,3', type=str)
parser.add_argument('--mixing-set',type=str,default='fractals',help='Mixing set directory.')
parser.add_argument('--use_300k',action='store_true',default=False,help='use 300K random images as aug data')
parser.add_argument('--model','-m',type=str,default='wrn',help='Choose architecture.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr',type=float,default=0.1,help='Initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=128)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay','-wd',type=float,default=0.0005,help='Weight decay (L2 penalty).')
# WRN Architecture options
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=4, type=int, help='Widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='Dropout probability')
# PixMix options
parser.add_argument('--beta',default=3,type=int,help='Severity of mixing')
parser.add_argument('--k',default=4,type=int,help='Mixing iterations')
parser.add_argument('--aug-severity',default=3,type=int,help='Severity of base augmentation operators')
parser.add_argument('--all-ops','-all',default=True,action='store_true',help='Turn on all augmentation operations (+brightness,contrast,color,sharpness).')
# Checkpointing options
parser.add_argument('--save','-s',type=str,default='./snapshots',help='Folder to save checkpoints.')
parser.add_argument('--resume','-r',type=str,default='',help='Checkpoint path for resume / test.')
parser.add_argument('--evaluate', action='store_true', help='Eval only.')
parser.add_argument('--print-freq',type=int, default=50,help='Training loss print frequency (batches).')
# Acceleration
parser.add_argument('--num-workers',type=int,default=4,help='Number of pre-fetching threads.')

parser.add_argument('--use_sam', default=True,type = bool)
parser.add_argument("--minimizer", default='ASAM', type=str, help="ASAM or SAM.")
parser.add_argument("--rho", default=1.2, type=float, help="Rho for ASAM.")
parser.add_argument("--eta", default=0.0, type=float, help="Eta for ASAM.")
parser.add_argument("--add", default=0.0, type=float, help="add")

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
print('*'*10, 'Using GPU:',args.gpus)

import pixmix_utils as utils
from asam import *
import numpy as np
from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet
from models.vit import *
from models.vgg import *
from models.densenet import *
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import torchvision

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

CBAR_CORRUPTIONS = [
    "blue_noise_sample", "brownish_noise", "checkerboard_cutout", 
    "inverse_sparkles", "pinch_and_twirl", "ripple", "circular_motion_blur", 
    "lines", "sparkles", "transverse_chromatic_abberation"]

NUM_CLASSES = 100 if args.dataset == 'cifar100' else 10

def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))


def pixmix(orig, mixing_pic, preprocess):
  
  mixings = utils.mixings
  tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
  if np.random.random() < 0.5:
    mixed = tensorize(augment_input(orig))
  else:
    mixed = tensorize(orig)
  
  for _ in range(np.random.randint(args.k + 1)):
    
    if np.random.random() < 0.5:
      aug_image_copy = tensorize(augment_input(orig))
    else:
      aug_image_copy = tensorize(mixing_pic)

    mixed_op = np.random.choice(mixings)
    mixed = mixed_op(mixed, aug_image_copy, args.beta)
    mixed = torch.clip(mixed, 0, 1)

  return normalize(mixed)

def augment_input(image):
  aug_list = utils.augmentations_all if args.all_ops else utils.augmentations
  op = np.random.choice(aug_list)
  return op(image.copy(), args.aug_severity)

class RandomImages300K(torch.utils.data.Dataset):
    def __init__(self, file, transform):
        self.dataset = np.load(file)
        self.transform = transform

    def __getitem__(self, index):
        img = self.dataset[index]
        return self.transform(img), 0

    def __len__(self):
        return len(self.dataset)

class PixMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform PixMix."""

  def __init__(self, dataset, mixing_set, preprocess):
    self.dataset = dataset
    self.mixing_set = mixing_set
    self.preprocess = preprocess

  def __getitem__(self, i):
    x, y = self.dataset[i]
    rnd_idx = np.random.choice(len(self.mixing_set))
    mixing_pic, _ = self.mixing_set[rnd_idx]
    return pixmix(x, mixing_pic, self.preprocess), y

  def __len__(self):
    return len(self.dataset)


def train(net, train_loader, optimizer, scheduler, minimizer=None):
  """Train for one epoch."""
  net.train()
  loss_ema = 0.
  for i, (images, targets) in enumerate(train_loader):

    # scheduler.step()
    images = images.cuda()
    targets = targets.cuda()

    outputs = net(images)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()
    minimizer.ascent_step()

    # Descent Step
    outputs = net(images)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()
    minimizer.descent_step()

    loss_ema = loss_ema * 0.9 + float(loss) * 0.1
    # if i % args.print_freq == 0:
    #   print('Train Loss {:.3f}'.format(loss_ema))

  return loss_ema


def test(net, test_loader, adv=None):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      # adversarial
      # if adv:
      #   images = adv(net, images, targets)
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader), total_correct / len(
      test_loader.dataset)


def test_c(net, test_data, base_path):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = []
  corrs = CBAR_CORRUPTIONS if 'Bar' in base_path else CORRUPTIONS
  for corruption in corrs:
    # Reference to original data is mutated
    test_data.data = np.load(base_path + corruption + '.npy')
    test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loss, test_acc = test(net, test_loader)
    corruption_accs.append(test_acc)
    print('{}\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        corruption, test_loss, 100 - 100. * test_acc))

  return np.mean(corruption_accs)

def normalize_l2(x):
  """
  Expects x.shape == [N, C, H, W]
  """
  norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
  norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
  return x / norm

class PGD(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, grad_sign=True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        # unnormalize
        bx = (bx+1)/2

        adv_bx = bx.detach()
        adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)

        for i in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits = model(adv_bx * 2 - 1)
                loss = F.cross_entropy(logits, by, reduction='sum')
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            if self.grad_sign:
                adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_bx = adv_bx.detach() + self.step_size * grad

            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(0, 1)

        return adv_bx*2-1

def main():
  torch.manual_seed(1)
  np.random.seed(1)
  torch.cuda.manual_seed(1)

  # Load datasets
  train_transform = transforms.Compose(
      [transforms.RandomHorizontalFlip(),
          transforms.RandomCrop(32, padding=4)
       ])
  mixing_set_transform = transforms.Compose(
      [transforms.Resize(36), 
       transforms.RandomCrop(32)])

  to_tensor = transforms.ToTensor()
  # normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
  #                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
  normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
  test_transform = transforms.Compose(
      [transforms.ToTensor(), normalize])

  if args.dataset == 'cifar10':
    train_data = datasets.CIFAR10('../cifar_data', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR10('../cifar_data', train=False, transform=test_transform, download=True)
    # base_c_path = os.path.join(args.data_path, 'cifar/CIFAR-10-C/')
    # base_c_bar_path = os.path.join(args.data_path, 'cifar/CIFAR-10-C-Bar/')
    num_classes = 10
  else:
    train_data = datasets.CIFAR100('../cifar_data', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR100('../cifar_data', train=False, transform=test_transform, download=True)
    # base_c_path = os.path.join(args.data_path, 'cifar/CIFAR-100-C/')
    # base_c_bar_path = os.path.join(args.data_path, 'cifar/CIFAR-100-C-Bar/')
    num_classes = 100

  if args.use_300k:
    mixing_set = RandomImages300K(file='300K_random_images.npy', transform=transforms.Compose(
      [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip()]))
  else:
    mixing_set = datasets.ImageFolder(args.mixing_set, transform=mixing_set_transform)
  print('train data',args.dataset, 'image sie:', len(train_data))
  print('aug_size', len(mixing_set))

  train_data = PixMixDataset(train_data, mixing_set, {'normalize': normalize, 'tensorize': to_tensor})
  # train_data_pixmix = test_transform(train_data_pixmix)


  # Fix dataloader worker issue
  # https://github.com/pytorch/pytorch/issues/5059
  def wif(id):
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))

  train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=True,
      worker_init_fn=wif)

  test_loader = torch.utils.data.DataLoader(
      test_data,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True)
  # train_loader = torch.utils.data.DataLoader(
  #     train_data,
  #     batch_size=args.eval_batch_size,
  #     shuffle=False,
  #     num_workers=args.num_workers,
  #     pin_memory=True)

  # Create model
  if args.model == 'vgg':
      net = VGG("VGG19", 32, num_classes)
  elif args.model == 'dense':
      net = DenseNet121(32, num_classes)
  elif args.model == 'wrn':
      net = WideResNet(40, num_classes, 4, 0.3)
  elif args.model == 'resnet50':
      # net = resnext29(num_classes=num_classes)
      net = torchvision.models.resnet50(weights="IMAGENET1K_V2")
      print('Loading ', args.model)
      n_features = net.fc.in_features
      fc = torch.nn.Linear(n_features, num_classes)
      fc.weight.data.normal_(0, 0.005)
      fc.bias.data.fill_(0.1)
      net.fc = fc
  elif args.model == 'vit':
      net = ViT(image_size=32, patch_size=4, num_classes=num_classes,
                dim=192, depth=12, heads=3, mlp_dim=768, dropout=0.1, emb_dropout=0.1)

  # Distribute model across all visible GPUs
  print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in net.parameters()])))
  net = torch.nn.DataParallel(net).cuda()
  cudnn.benchmark = True

  # initialize adversary
  adversary = PGD(epsilon=2. / 255, num_steps=20, step_size=0.5 / 255).cuda()

  start_epoch = 0

  if args.resume:
      if os.path.isfile(args.resume):
          checkpoint = torch.load(args.resume)
          start_epoch = checkpoint['epoch'] + 1
          best_acc = checkpoint['best_acc']
          net.load_state_dict(checkpoint['state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer'])
          print('Model restored from epoch:', start_epoch)

  if args.evaluate:
      # Evaluate clean accuracy first because test_c mutates underlying data
      test_loss, test_acc = test(net, test_loader)
      print('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
          test_loss, 100 - 100. * test_acc))

      # adv_test_loss, adv_test_acc = test(net, test_loader, adv=adversary)
      # print('Adversarial\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
      #     adv_test_loss, 100 - 100. * adv_test_acc))

      # test_c_acc = test_c(net, test_data, base_c_path)
      # print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))
      return

  if args.opti == 'RMS':
      optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.decay, momentum=0.9)
  # elif args.opti == 'SGDM':
  #     optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
  elif args.opti == 'SGD':
      optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
  elif args.opti == 'adam':
      optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)
  elif args.opti == 'adamW':
      optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.2)

  if args.use_sam:
      if args.minimizer == "Layer_ASAM":
          minimizer = eval(args.minimizer)(optimizer, net, layer_rho=init_layer_rho(args, model), eta=args.eta)
      else:
          minimizer = ASAM(optimizer, net, rho=args.rho, eta=args.eta)
  else:
      minimizer = None

  # print(args.use_sam)
  # print(minimizer)
  # e
  save_path = 'snapshots/' + args.model + '_' + args.opti + '_' + str(args.batch_size) + '_' + str(args.lr) + '_ASAM'
  os.makedirs(save_path, exist_ok=True)
  log_path = os.path.join(save_path, args.dataset + '_' + args.model + '_training_log.csv')

  with open(log_path, 'w') as f:
    f.write('epoch,LR,train_loss,test_loss,test_error(%)\n')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'''Starting training:      
                      Model name:      {args.model}
                      Epochs:          {args.epochs}
                      Batch size:      {args.batch_size}
                      Learning rate:   {args.lr}
                      Training size:   {len(train_loader.dataset)}
                      Beta: {args.beta}
                      training data:    {args.dataset}
                      save_path:    {save_path}
                  ''')

  best_acc = 0
  scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer,
      lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
          step,
          args.epochs * len(train_loader),
          1,  # lr_lambda computes multiplicative factor
          1e-6 / args.lr))
  print('Beginning training from epoch:', start_epoch + 1)

  for epoch in range(start_epoch, args.epochs):
    # begin_time = time.time()

    # r = np.random.rand(1)
    # if r <= 0.5:
    #     train_loss_ema = train(net, train_loader_pixmix, optimizer, scheduler)
    # else:
    start = time.time()
    train_loss_ema = train(net, train_loader, optimizer, scheduler, minimizer=minimizer)
    print('Using time',time.time()-start)
    test_loss, test_acc = test(net, test_loader)

    adjust_lr(optimizer, epoch)

    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    checkpoint = {
        'epoch': epoch,
        'dataset': args.dataset,
        'model': args.model,
        'state_dict': net.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }

    torch.save(checkpoint, save_path + '/checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(save_path + '/checkpoint.pth.tar', save_path + '/model_best.pth.tar')

    with open(log_path, 'a') as f:
      f.write('%03d,%0.6f,%0.6f,%0.5f,%0.2f\n' % (
          (epoch + 1),
          optimizer.param_groups[0]['lr'],
          train_loss_ema,
          test_loss,
          100 - 100. * test_acc,
      ))

    print(
        'Epoch {0:3d} | Train Loss {2:.4f} | Test Loss {3:.3f} |'
        ' Test Error {4:.2f} | {1}'
        .format((epoch + 1), 'Pixmix', train_loss_ema,
                test_loss, 100 - 100. * test_acc))

    if (epoch+1) % 10 == 0:
        torch.save(checkpoint, save_path + '/epoch_' + str(epoch+1) + '.pth.tar')

  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  logging.info(f'''Starting training:      
                      Model name:      {args.model}
                      Epochs:          {args.epochs}
                      Batch size:      {args.batch_size}
                      Learning rate:   {args.lr}
                      Training size:   {len(train_data)}
                      Test size: {len(test_data)}
                      training data:    {args.dataset}
                      saving weights to:  {save_path}
                  ''')
  # _, adv_test_acc = test(net, test_loader, adv=adversary)
  # print('Adversarial Test Error: {:.3f}\n'.format(100 - 100. * adv_test_acc))
  
  # test_c_acc = test_c(net, test_data, base_c_path)
  # print('Mean C Corruption Error: {:.3f}\n'.format(100 - 100. * test_c_acc))

  # test_c_bar_acc = test_c(net, test_data, base_c_bar_path)
  # print('Mean C-Bar Corruption Error: {:.3f}\n'.format(100 - 100. * test_c_bar_acc))

  # print('Mean Corruption Error: {:.3f}\n'.format(100 - 100. * (15*test_c_acc + 10*test_c_bar_acc)/25))

  # with open(log_path, 'a') as f:
  #   f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
  #           (args.epochs + 1, 0, 0, 0, 100 - 100 * test_c_acc))
def adjust_lr(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
  main()

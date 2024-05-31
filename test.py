# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py
import  numpy as np
import argparse
import os
import shutil
import time
import logging
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
# import resnet as RN
# import pyramidnet as PYRM
import torchvision
from dataloader import *
from models.vit import *
from models.vgg import *
from models.densenet import *
import warnings
from models.wideresnet import WideResNet

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Test')
parser.add_argument('--model', default='wrn', type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--dataset', dest='dataset', default='cifar100', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false', default=True,
                    help='to print the status at every iteration')
parser.add_argument('--load_pth', '-lp',default='/set/your/model/path', type=str, metavar='PATH')
parser.add_argument('--gpus', default='0,1,2,3', type=str)


parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

CORRUPTIONS = ["gaussian_noise", "shot_noise", "speckle_noise", "impulse_noise",
               "defocus_blur", "gaussian_blur", "motion_blur", "zoom_blur",'glass_blur',
               "snow", "fog","brightness", "contrast", "elastic_transform", "pixelate",
               "jpeg_compression", "spatter", "saturate", "frost"]

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
print('*'*10, 'Using GPU:',args.gpus)

CBAR_CORRUPTIONS = [
    "blue_noise_sample", "brownish_noise", "checkerboard_cutout",
    "inverse_sparkles", "pinch_and_twirl", "ripple", "circular_motion_blur",
    "lines", "sparkles", "transverse_chromatic_abberation"]

def main():
    global args, best_err1, best_err5
    args = parser.parse_args()

    if args.dataset.startswith('cifar'):
        # if 'pixmix' in args.load_pth or 'augmix' in args.load_pth:
        normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
        # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                                  (0.2023, 0.1994, 0.2010))
        # normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
        # else:
        #     normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        #                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        #

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        checkpoint = torch.load(args.load_pth)
        dataset = checkpoint['dataset']

        if dataset == 'cifar100':
            val_data = datasets.CIFAR100('./cifar_data', train=False, transform=transform_test)
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 100
        elif dataset == 'cifar10':
            val_data = datasets.CIFAR10('./cifar_data', train=False, transform=transform_test)
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))

    elif args.dataset == 'imagenet':

        valdir = os.path.join('/home/data/ILSVRC/val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        numberofclass = 1000

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    model_name = args.load_pth.split('/')[2].split('_')[0]

    print("=> creating model '{}'".format(model_name))

    if model_name == 'vgg':
        model = VGG("VGG19", 32, numberofclass)
    elif model_name == 'dense':
        model = DenseNet121(32, numberofclass)
    elif model_name == 'wrn':
        model = WideResNet(40, numberofclass, 4, 0.3)
        # model = WideResNet(34, numberofclass)

    elif model_name == "vit":
        model = ViT(image_size=32, patch_size=4, num_classes=numberofclass,
                    dim=192, depth=12, heads=3, mlp_dim=768, dropout=0.1, emb_dropout=0.1)
    else:
        raise Exception('unknown network architecture: {}'.format(model_name))

    model = torch.nn.DataParallel(model).cuda()


    if os.path.isfile(args.load_pth):
        print("=> loading checkpoint '{}'".format(args.load_pth))
        checkpoint = torch.load(args.load_pth)
        print('=> Data is', checkpoint['dataset'])
        print('=> epoch is', checkpoint['epoch'])
        dataset = checkpoint['dataset']
        # print(checkpoint['state_dict'])
        # e
        # print(checkpoint.keys())
        model.load_state_dict(checkpoint['state_dict'],strict=False)

        # print("=> loaded checkpoint '{}'".format(args.load_pth))
    else:
        raise Exception("=> no checkpoint found at '{}'".format(args.load_pth))

    # print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True
    tmp = args.load_pth.split('/')[0].split('_')[0]
    all_name = args.load_pth.split('/')[2]
    print(tmp)

    if 'best' in args.load_pth:
        os.makedirs(os.path.join('results', model_name, 'best'), exist_ok=True)
        log_path = 'results/' + model_name + '/best/' + tmp + '_' + dataset + '_' + all_name + '.csv'
        if 'Clean' in args.load_pth:
            log_path = 'results/'+ model_name + '/best/' + 'Clean_'+ dataset + '_' + all_name + '.csv'
    else:
        os.makedirs(os.path.join('results', model_name, 'last'), exist_ok=True)
        log_path = 'results/' + model_name + '/last/' + tmp + '_' + dataset + '_' + all_name + '.csv'
        if 'Clean' in args.load_pth:
            log_path = 'results/' + model_name + '/last/' + 'Clean_' + dataset + '_' + all_name + '.csv'
    #
    with open(log_path, 'w') as f:
        f.write('type, Error\n')

    # logging.info(f'Saving to... {log_path}')
    # evaluate on validation set
    err1, err5, val_loss = validate(val_loader, model, criterion)
    print('*' * 10, args.load_pth)
    print('*'*20)
    print('Clean Error(top-1 and 5 error):', err1, err5)

    with open(log_path, 'a') as f:
        f.write('%s,%0.2f\n' % ('Clean', err1))

    #
    start = time.time()
    if dataset == 'cifar100':
        avg_loss, avg_err = test_cc(model, val_data, log_path,'./CIFAR-100-C/')
    else:
        avg_loss, avg_err = test_cc(model, val_data, log_path, './CIFAR-10-C/')
    print('*' * 20)
    print('Error (Corrupted)', avg_err)
    print('Loss (Corrupted)', avg_loss)
    print('Using time:', time.time() - start)
    with open(log_path, 'a') as f:
        f.write('%s,%0.2f\n' % ('Avg_Err', avg_err))
        f.write('%s,%0.2f\n' % ('Avg_Loss', avg_loss))

    logging.info(f'Saving to... {log_path}')

    # start = time.time()
    # err = test_c(model, './CIFAR-100-C/')
    # print('Error (Corrupted)', err)
    # print('Using time:', time.time() - start)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    Acc_c = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = Top_err(output.data, target, topk=(1, 5))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))
        # Acc_c.update(acc.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0 and args.verbose == True:
        #     print('Test (on val set): [{0}/{1}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
        #           'Top 5-err {top5.val:.4f} ({top5.avg:.4f})\t'.format(
        #         i, len(val_loader), batch_time=batch_time, loss=losses,
        #         top1=top1, top5=top5))



    return top1.avg, top5.avg, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Top_err(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # output = [64*100]
    # target = [64]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res

def test(net, test_loader, adv=None):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      # adversarial
      if adv:
        images = adv(net, images, targets)
      # print(images[0,0,:3,:3])
      # e
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  # print(total_correct, len(test_loader.dataset))
  # e
  return total_loss / len(test_loader), total_correct / len(test_loader.dataset)

def test_cc(net, test_data, log_path, base_path="./data/CIFAR-100-C/"):
  """Evaluate network on given corrupted dataset."""
  net.eval()
  corruption_err = []
  corruption_loss = []
  corrs = CBAR_CORRUPTIONS if 'Bar' in base_path else CORRUPTIONS
  for corruption in corrs:
    # Reference to original data is mutated
    test_data.data = np.load(base_path + corruption + '.npy')  # pixle range = [0,255]
    # print(test_data.data[0,0,:3,:])
    # e
    test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True)

    # img,_ = next(iter(test_loader))
    # print(img[0, 0, :3, :3])
    # e
    test_loss, test_acc = test(net, test_loader)
    test_err = 100 - 100. * test_acc
    # e
    corruption_err.append(test_err)
    corruption_loss.append(test_loss)
    print('{}\t | Test Error {:.3f}'.format(
        corruption, test_err))
    with open(log_path, 'a') as f:
        # f.write('%s,%0.2f\n' % (corruption, test_loss))
        f.write('%s,%0.2f\n' % (corruption, test_err))

  return np.mean(corruption_loss), np.mean(corruption_err)

def test_c(model,base_path):
  """Evaluate network on given corrupted dataset."""
  model.eval()
  correct = 0

  corruption_accs = []
  corrs = CBAR_CORRUPTIONS if 'Bar' in base_path else CORRUPTIONS
  corruption_acc_dict = {}
  for corruption in corrs:
      total = 0
      avg_acc = 0.0
      # print()
      dataloader = torch.utils.data.DataLoader(CIFAR100C(corruption), batch_size=args.batch_size, shuffle=False,
                                               num_workers=8)
      with torch.no_grad():
          for batch_idx, (inputs, targets) in enumerate(dataloader):
              inputs, targets = inputs.cuda(), targets.cuda()
              # print(inputs[0, 0, :3, :3])
              # e
              outputs = model(inputs)
              _, predicted = outputs.max(1)
              total += targets.size(0)
              correct += predicted.eq(targets).sum().item()

      # print(total,correct)
      # e
      acc = 100. * correct / total
      corruption_acc_dict[corruption] = 100 - acc
      print('{}\t| Test Error {:.3f}'.format(
          corruption, 100 - acc))
      avg_acc += acc

  corruption_acc_dict["avg"] = avg_acc / len(corrs)

  return corruption_acc_dict

if __name__ == '__main__':
    main()

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import logging

import torch
import torch.nn as nn
import torch.nn.init as init

import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from dataloader import CIFAR10C, CIFAR100C


def load_sd(model_path):
    sd = torch.load(model_path)
    if "net" in sd.keys():
        sd = sd["net"]
    elif "state_dict" in sd.keys():
        sd = sd["state_dict"]
    elif "model" in sd.keys():
        sd = sd["model"]

    return sd


def evaluate_corruption(args, model):
    model.eval()
    avg_acc = 0.0

    corruptions = [
        "gaussian_noise",
        "shot_noise",
        "speckle_noise",
        "impulse_noise",
        "defocus_blur",
        "gaussian_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
        "spatter",
        "saturate",
        "frost"
    ]

    corruption_acc_dict = {}
    for cname in corruptions:
        correct = 0
        total = 0
        
        if args.dataset == "CIFAR10":
            dataloader = torch.utils.data.DataLoader(CIFAR10C(cname), batch_size=args.batch_size, shuffle=False, num_workers=8)
        elif args.dataset == "CIFAR100":
            dataloader = torch.utils.data.DataLoader(CIFAR100C(cname), batch_size=args.batch_size, shuffle=False, num_workers=8)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        corruption_acc_dict[cname] = acc
        avg_acc += acc
        print(corruption_acc_dict)

    corruption_acc_dict["avg"] = avg_acc / len(corruptions)


    return corruption_acc_dict


def evaluate(args, model, dataloader, criterion):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return loss, acc


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean).to("cuda"))
        self.register_buffer('std', torch.Tensor(std).to("cuda"))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


def register_forward_hook(net, hook_input=True):
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            if hook_input:
                # The input hooked is a tuple, input[0] is the layer input.
                activations[name] = input[0]
            else:
                activations[name] = output

        return hook

    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # print(name)
            module.register_forward_hook(get_activation(name))

    # if hook_input:
    #     for k, v in activations.items():
    #         activations[k] = v[0]

    return activations


def register_backward_hook(net):
    grads = {}

    def get_grads(name):
        def hook(module, grad_input, grad_output):
            grads[name] = grad_input

        return hook

    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # print(name)
            module.register_full_backward_hook(get_grads(name))

    # if hook_input:
    #     for k, v in activations.items():
    #         activations[k] = v[0]

    return grads


# def find_last_layer_name(net):
#     return "module.linear"
#     # return "fc"


def parse_model_path(model_path):
    # "./results/ResNet18_CIFAR10/checkpoint/SGDM_lr=0.1_bs=128_wd=0.0_do=0.0",
    super_params = model_path.split("/")

    model_dataset = super_params[2].split("_")
    model = model_dataset[0]
    dataset = model_dataset[1]

    hyper_params = super_params[4].split("_")
    optim = hyper_params[0]
    lr = float(hyper_params[1].split("=")[-1])
    bs = int(hyper_params[2].split("=")[-1])
    wd = float(hyper_params[3].split("=")[-1])
    do = float(hyper_params[4].split("=")[-1])

    if model == "ViT":
        patch = 4
    else:
        patch = 0

    if model == "ViT-timm":
        resize = 224
    else:
        resize = None

    return {
        "model": model,
        "dataset": dataset,
        "optim": optim,
        "lr": lr,
        "bs": bs,
        "wd": wd,
        "do": do,
        "patch": patch,
        "resize": resize
    }


def plot_curve(x, y, save_dir, show_y=True):
    plt.plot(x, y)
    if show_y:
        for i, (xi, yi) in enumerate(zip(x, y)):
            if i % 2 == 0:
                plt.text(xi, yi, np.round(yi, 7), fontsize=4)
    plt.savefig(save_dir)
    plt.close()


def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def get_all_trained_model_params(path):
    trained_params_list = []

    for (root, dirs, files) in os.walk(path):
        # print(root, dirs, files)
        if len(files) > 0:
            for my_file in files:
                if my_file.find(".pth") != -1:
                    trained_params_list.append(root + "/" + my_file)
    # print(trained_params_list)
    # exit()
    return trained_params_list

from copy import deepcopy

from utils import *
from dataloader import *
from model import create_model
from optimizer import *
import torch.optim as optim
from asam import *
import torch
import torchvision


def cal_layer_rho(args, model):
    num_layers = 0
    for name, _ in model.named_parameters():
        num_layers += 1

    cur_num = 0
    layer_rho = {}
    for name, _ in model.named_parameters():
        layer_rho[name] = args.rho + cur_num / num_layers * args.add
        cur_num += 1

    return layer_rho

def load_model(model_name, num_classes, device, pre_trained=True):

	if pre_trained:
		if model_name == 'vgg19':
			model = torchvision.models.vgg19(pretrained=True)
			n_features = model.classifier[6].in_features
			fc = torch.nn.Linear(n_features, num_classes)
			fc.weight.data.normal_(0, 0.005)
			fc.bias.data.fill_(0.1)
			model.classifier[6] = fc
		elif model_name == 'resnet50':
			model = torchvision.models.resnet50(pretrained=True)
			# print(model)
			n_features = model.fc.in_features
			fc = torch.nn.Linear(n_features, num_classes)
			fc.weight.data.normal_(0, 0.005)
			fc.bias.data.fill_(0.1)
			model.fc = fc
			# print(model)
		elif model_name == 'vit':
			model = torchvision.models.vit_b_16(pretrained=True)
			n_features = model.heads.head.in_features
			fc = torch.nn.Linear(n_features, num_classes)
			fc.weight.data.normal_(0, 0.005)
			fc.bias.data.fill_(0.1)
			model.heads.head = fc

	if device == 'cuda':
		model = torch.nn.DataParallel(model)
	return model

def get_optimizer(model_name, model, lr,momentum):
    learning_rate = lr

    # 微调策略：最后一层全连接的学习率与前面层的学习率不同。
    if model_name == 'vgg19':
        param_group = [
            {'params': model.features.parameters(), 'lr': learning_rate}]
        for i in range(6):
            param_group += [{'params': model.classifier[i].parameters(),
                             'lr': learning_rate}]
        param_group += [{'params': model.classifier[6].parameters(),
                         'lr': learning_rate * 10}]
    elif model_name == 'resnet50':
        param_group = []
        for k, v in model.named_parameters():
            if not k.__contains__('fc'):
                param_group += [{'params': v, 'lr': learning_rate}]
            else:
                param_group += [{'params': v, 'lr': learning_rate * 10}]
    elif model_name == 'vit':
        param_group = []
        for k, v in model.named_parameters():
            if not k.__contains__('heads.head'):
                param_group += [{'params': v, 'lr': learning_rate}]
            else:
                param_group += [{'params': v, 'lr': learning_rate * 10}]
    optimizer = optim.SGD(param_group, momentum=momentum)
    return optimizer


def train(args, model, dataloader, optimizer, criterion, minimizer=None):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        if args.use_sam:
            # Ascent Step
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            minimizer.ascent_step()

            # Descent Step
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            minimizer.descent_step()
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

        train_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(i, len(dataloader), 'Loss: {:.2f} | Acc: {:.2f}%'.format(train_loss/total, 100.*correct/total, correct, total))

    return train_loss / total, correct / total * 100


def main():

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--dataset', default="CIFAR10", type=str, help="dataset")
    parser.add_argument('--num_classes', default=10, type=int, help='num classes')
    parser.add_argument('--input_size', default=32, type=int, help='input_size')
    parser.add_argument('--patch', default=4, type=int, help='num patch (used by vit)')

    parser.add_argument('--device', default="cuda", type=str, help='device')

    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_scheduler', default="step", choices=["step", 'cosine'])

    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGDM')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=float, help='lr_decay_gamma')
    parser.add_argument('--wd', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--epochs', default=200, type=int, help='num of epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')

    parser.add_argument('--optim', default="SGDM", type=str, help="optimizer")
    parser.add_argument('--model', default="resnet50", type=str, help='model used',choices=['resnet50','vgg19','vit'])
    parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--use_sam', action='store_true')
    parser.add_argument("--minimizer", default='ASAM', type=str, help="ASAM or SAM.")
    parser.add_argument("--rho", default=1.2, type=float, help="Rho for ASAM.")
    parser.add_argument("--eta", default=0.0, type=float, help="Eta for ASAM.")
    parser.add_argument("--add", default=0.0, type=float, help="add")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    suffix = '{}_lr={}_wd={}_epochs={}'.format(args.optim, args.lr, args.wd, args.epochs)

    if args.use_sam:
        suffix = "SAM_{:.2f}_".format(args.rho)+suffix

    model_save_dir = './results/{}_{}/checkpoint/'.format(args.model, args.dataset) + suffix + "/"
    curve_plot_dir = './results/{}_{}/curve_plot/'.format(args.model, args.dataset) + suffix + "/"

    logger = create_logger('./logs/{}_{}_{}.log'.format(args.model, args.dataset, suffix))
    logger.info(args)

    for path in [model_save_dir, curve_plot_dir]:
        if not os.path.isdir(path):
            os.makedirs(path)

    # create dataloader
    logger.info('==> Preparing data and create dataloaders...')

    trainloader, _, testloader = create_dataloader(args.dataset, args.batch_size, use_val=False, transform_dict=None)

    logger.info('==> Building dataloaders...')
    logger.info(args.dataset)

    # create model
    logger.info('==> Building model...')
    model = load_model(args.model, args.num_classes, args.device)
    logger.info(args.model)

    logger.info('==> Building optimizer and learning rate scheduler...')
    optimizer = get_optimizer(args.model, model, args.lr, args.momentum)
    logger.info(optimizer)

    if args.use_sam:
        if args.minimizer == "Layer_ASAM":
            minimizer = eval(args.minimizer)(optimizer, model, layer_rho=cal_layer_rho(args, model), eta=args.eta)
        else:
            minimizer = eval(args.minimizer)(optimizer, model, rho=args.rho, eta=args.eta)
    else:
        minimizer = None

    lr_decays = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    scheduler = create_scheduler(args, optimizer, lr_decays=lr_decays)
    logger.info(scheduler)

    criterion = nn.CrossEntropyLoss()

    init_sd = deepcopy(model.state_dict())
    torch.save(init_sd, model_save_dir + "init_params.pth")

    for epoch in range(start_epoch, start_epoch + args.epochs):

        logger.info("==> Epoch {}".format(epoch))
        logger.info("==> Training...")
        train_loss, train_acc = train(args, model, trainloader, optimizer, criterion, minimizer=minimizer)
        logger.info("==> Train loss: {:.2f}, train acc: {:.2f}%".format(train_loss, train_acc))

        logger.info("==> Testing...")
        test_loss, test_acc = evaluate(args, model, testloader, criterion)

        logger.info("==> Test loss: {:.2f}, test acc: {:.2f}%".format(test_loss, test_acc))

        state = {
            'model': model.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        if test_acc > best_acc:
            best_acc = test_acc
            params = "best_params.pth"
            logger.info('==> Saving best params...')
            torch.save(state, model_save_dir + params)
        else:
            if epoch % 2 == 0:
                params = "epoch{}_params.pth".format(epoch)
                logger.info('==> Saving checkpoints...')
                torch.save(state, model_save_dir + params)

        scheduler.step()

    checkpoint = torch.load(model_save_dir + "best_params.pth")
    model.load_state_dict(checkpoint["model"])
    print(evaluate_corruption(args, model))


if __name__ == "__main__":
    main()







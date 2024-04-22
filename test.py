from utils import *
from dataloader import *
from model import create_model


def main():

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--dataset', default="CIFAR100", type=str, help="dataset")
    parser.add_argument('--num_classes', default=100, type=int, help='num classes')
    parser.add_argument('--input_size', default=32, type=int, help='input_size')
    parser.add_argument('--patch', default=4, type=int, help='num patch (used by vit)')

    parser.add_argument('--device', default="cuda", type=str, help='device')

    parser.add_argument('--batch_size', default=128, type=int, help='batch size')

    parser.add_argument('--model', default="WideResNet34", type=str, help='model used')
    parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')

    args = parser.parse_args()

    # args.resume = "/home/wjd/ZKJ/OOD_Sharpness/results/ResNet18_CIFAR10/checkpoint/SGDM_lr=0.1_wd=0.0005_epochs=200/best_params.pth"
    # args.resume = "/home/wjd/ZKJ/OOD_Sharpness/results/ResNet18_CIFAR10/checkpoint/SAM_5.00_SGDM_lr=0.1_wd=0.0005_epochs=200/best_params.pth"

    # args.resume = "/home/wjd/ZKJ/Bag-of-Tricks-for-AT/trained_models/ResNet18_piecewise_eps8_bs128_maxlr0.1_attackiters10_wd0.0005/model_test_best.pth"
    # args.resume = "/home/wjd/ZKJ/Finetune/results/ResNet18_CIFAR10/checkpoint/Finetune_SGDM_lr=0.001_wd=0.0005_epochs=10_None/finetune_0.8_params.pth"
    # args.resume = "/mnt/data1/ZKJ_data/ResNet50_CIFAR100_baseline.pth"
    # args.resume = "/mnt/data1/ZKJ_data/ResNet50_CIFAR100_SAM.pth"
    # args.resume = "/mnt/data1/ZKJ_data/WRN34_CIFAR100_baseline.pth"
    args.resume = "/mnt/data1/ZKJ_data/WRN34_CIFAR100_SAM.pth"

    # create model
    model = create_model(args.model, args.input_size, args.num_classes, args.device, args.patch, args.resume)
    corruption_acc_dict = evaluate_corruption(args, model)
    print(corruption_acc_dict)


if __name__ == "__main__":
    main()

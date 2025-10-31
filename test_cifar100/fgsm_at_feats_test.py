import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from fat.fgsm_at_feats import FGSM_AT_Feats
from models import MyResNet18
from utils.dataset import train_transform, test_transform


def fgsm_at_feats_test(seed=0, device=0, alpha=0.6, name="test", label_smoothing=0.5, T=8.0):
    device = torch.device(
        "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"
    )

    train_set = CIFAR100(
        "/home/chenyu/ADV/data", train=True, download=True, transform=train_transform
    )
    test_set = CIFAR100(
        "/home/chenyu/ADV/data", train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=False, num_workers=4
    )

    model = MyResNet18(num_classes=100).to(device)

    opt = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    num_class = 100
    scheduler = MultiStepLR(opt, milestones=[100, 105], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    fgsm_uap = FGSM_AT_Feats(
        model,
        device=device,
        seed=seed,
        log_dir="./log_cifar100/",
        num_class=num_class,
        name=name,
    )
    fgsm_uap.train(
        opt,
        scheduler,
        train_loader,
        test_loader,
        total_epoch=110,
        alpha=alpha,
        T=T,
        label_smoothing=label_smoothing,uap_num=200
    )

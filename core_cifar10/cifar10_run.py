import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from .core import FGSM_AT, FGSM_AT_Class, FGSM_AT_Feats
from models import MyResNet18
from utils.dataset import train_transform, test_transform


def cifar10_run(seed=0, device=0, alpha=0.6, name="test", T=8.0, label_smoothing=0.6, mode=0):
    device = torch.device(
        "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"
    )

    train_set = CIFAR10(
        "/home/xxxx/ADV/data", train=True, download=True, transform=train_transform
    )
    test_set = CIFAR10(
        "/home/xxxx/ADV/data", train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=False, num_workers=4
    )

    model = MyResNet18(num_classes=10).to(device)

    opt = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    num_class = 10
    scheduler = MultiStepLR(opt, milestones=[100, 105], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    if mode == 0:
        fat_trainer = FGSM_AT(
            model,
            device=device,
            seed=seed,
            log_dir="./log_cifar10/",
            num_class=num_class,
            name=name,
        )
    elif mode == 1:
        fat_trainer = FGSM_AT_Class(
            model,
            device=device,
            seed=seed,
            log_dir="./log_cifar10/",
            num_class=num_class,
            name=name,
        )
    elif mode == 2:
        fat_trainer = FGSM_AT_Feats(
            model,
            device=device,
            seed=seed,
            log_dir="./log_cifar10/",
            num_class=num_class,
            name=name,uap_num=50
        )
    fat_trainer.train(
        opt,
        scheduler,
        train_loader,
        test_loader,
        total_epoch=110,
        alpha=alpha,
        T=T,
        label_smoothing=label_smoothing,
    )

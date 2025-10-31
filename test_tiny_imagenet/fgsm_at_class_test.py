import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from models import MyPreActResNet18
from utils import set_seed
from utils import train_transform_tiny_imagenet, test_transform, TinyImageNet200

from fat.fgsm_at_class import FGSM_AT_Class


def fgsm_at_class_test(seed=0, device=0, alpha=0.6, name="test", T=8.0, label_smoothing=0.5):
    device = torch.device(
        "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"
    )

    train_set = TinyImageNet200(
        "/home/chenyu/ADV/data",
        train=True,
        download=True,
        transform=train_transform_tiny_imagenet,
    )

    test_set = TinyImageNet200(
        "/home/chenyu/ADV/data", train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=False, num_workers=4
    )

    model = MyPreActResNet18(num_classes=200).to(device)

    opt = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    num_class = 200
    scheduler = MultiStepLR(opt, milestones=[100, 105], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    fgsm_uap = FGSM_AT_Class(
        model,
        device=device,
        seed=seed,
        log_dir="./log_tiny_imagenet/",
        num_class=num_class,
        name=name,
    )
    fgsm_uap.train(
        opt,
        scheduler,
        train_loader,
        test_loader,
        total_epoch=110,
        label_smoothing=label_smoothing,
        alpha=alpha,
        image_shape=(3, 64, 64),
        T=T,
    )

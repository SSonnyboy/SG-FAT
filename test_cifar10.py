import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchattacks import MIFGSM, BIM, PGD, APGD, Square, FGSM
from autoattack import AutoAttack
from models import ResNet18

from utils.dataset import test_transform
from utils.logger import Logger
from utils.set_seed import set_seed
from utils.cw import *

from autoattack import AutoAttack


def get_attackers(model, device, log_path, ifAA):
    if ifAA == 1:
        return {
            "AA": AutoAttack(
                model,
                norm="Linf",
                eps=8.0 / 255,
                device=device,
                version="standard",
                log_path=log_path,
            ),
        }

    else:
        return {
            "PGD-10": PGD(model, eps=8.0 / 255, alpha=2.0 / 255, steps=10),
            "PGD-20": PGD(model, eps=8.0 / 255, alpha=2.0 / 255, steps=20),
            "PGD-50": PGD(model, eps=8.0 / 255, alpha=2.0 / 255, steps=50),
            "CW": CW_Linf(
                model,
                eps=8.0 / 255,
                alpha=2.0 / 255,
                steps=50,
                restarts=1,
                device=device,
            ),
        }


def evaluate(model, test_loader, attacker, device):
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = (
            attacker(images, labels)
            if hasattr(attacker, "__call__")
            else attacker.run_standard_evaluation(
                images, labels, bs=100, return_labels=False
            )
        )
        outputs = model(adv_images)
        _, pred = outputs.max(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return 100 * correct / total


def evaluate_all_attacks(model, test_loader, logger, device, log_path, ifAA):
    # Evaluate clean accuracy
    model.eval()
    clean_correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        clean_correct += (preds == labels).sum().item()
        total += labels.size(0)
    logger.log("Clean Accuracy: {:.2f}%".format(100 * clean_correct / total))

    # Evaluate all attacks
    attackers = get_attackers(model, device, log_path, ifAA)
    for name, attacker in attackers.items():
        if name == "AA":
            logger.log("Running AutoAttack...")
            X_all, y_all = [], []
            for images, labels in test_loader:
                X_all.append(images)
                y_all.append(labels)
            X_all = torch.cat(X_all).to(device)
            y_all = torch.cat(y_all).to(device)

            attacker.run_standard_evaluation(X_all, y_all, bs=100)
        else:
            acc = evaluate(model, test_loader, attacker, device)
            logger.log("{} Accuracy: {:.2f}%".format(name, acc))


def test_model(model_path, log_path, test_loader, device, ifAA):
    if os.path.exists(log_path):
        return

    set_seed(0)
    logger = Logger(log_path)
    model = ResNet18().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    evaluate_all_attacks(model, test_loader, logger, device, log_path, ifAA)

    logger.new_line()


def main(method, gpu, ifAA):
    test_loader = DataLoader(
        datasets.CIFAR10(
            "/home/chenyu/ADV/data", train=False, transform=test_transform
        ),
        batch_size=1000,
        shuffle=False,
        num_workers=4,
    )

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
    base_dir = "./log_cifar10/"
    method_names = []
    method_names.append(method)

    for method_name in method_names:
        method_path = os.path.join(base_dir, method_name)
        for time_dir in os.listdir(method_path):
            time_path = os.path.join(method_path, time_dir)
            if "seed" not in time_path:
                continue

            if ifAA == 1:
                # Test best model
                best_model = os.path.join(time_path, "best.pth")
                # best_log = os.path.join(time_path, method_name + "_best_all_attacks.log")
                best_log = os.path.join(time_path, method_name + "_best_AA.log")

                test_model(best_model, best_log, test_loader, device, ifAA)

                # Test last model
                last_model = os.path.join(time_path, "last.pth")
                # last_log = os.path.join(time_path, method_name + "_last_all_attacks.log")
                last_log = os.path.join(time_path, method_name + "_last_AA.log")

                test_model(last_model, last_log, test_loader, device, ifAA)
            else:
                # Test best model
                best_model = os.path.join(time_path, "best.pth")
                best_log = os.path.join(
                    time_path, method_name + "_best_all_attacks.log"
                )
                test_model(best_model, best_log, test_loader, device, ifAA)

                # Test last model
                last_model = os.path.join(time_path, "last.pth")
                last_log = os.path.join(
                    time_path, method_name + "_last_all_attacks.log"
                )
                test_model(last_model, last_log, test_loader, device, ifAA)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ifAA", type=int, default=0)

    args = parser.parse_args()
    main(args.method, args.gpu, args.ifAA)

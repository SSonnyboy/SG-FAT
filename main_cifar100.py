from core_cifar100 import cifar100_run
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--name", type=str, default="at-fat")
parser.add_argument("--alpha", type=float, default=0.6)
parser.add_argument("--T", type=float, default=8.0)
parser.add_argument("--ls", type=float, default=0.8)

parser.add_argument("--mode", type=int, default=0)  # 0: one uap, 1: class uap, 2: feat uap

args = parser.parse_args()

seed = args.seed
gpu = args.gpu
name = args.name
alpha = args.alpha
T = args.T
ls = args.ls    
mode = args.mode

cifar100_run(seed=seed, device=gpu, name=name, alpha=alpha, T=T, label_smoothing=ls, mode=mode)

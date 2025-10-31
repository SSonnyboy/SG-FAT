from test_cifar100 import fgsm_at_test, fgsm_at_feats_test, fgsm_at_class_test

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--name", type=str, default="at-fat")
parser.add_argument("--alpha", type=float, default=0.6)
parser.add_argument("--T", type=float, default=8.0)
parser.add_argument("--ls", type=float, default=0.5)


args = parser.parse_args()

# fgsm_at_test(
#     seed=args.seed,
#     device=args.gpu,
#     alpha=args.alpha,
#     name=args.name,
#     label_smoothing=args.ls,
#     T=args.T,
# )

# fgsm_at_class_test(
#     seed=args.seed,
#     device=args.gpu,
#     alpha=args.alpha,
#     name=args.name,
#     label_smoothing=args.ls,
#     T=args.T,
# )

fgsm_at_feats_test(
    seed=args.seed,
    device=args.gpu,
    alpha=args.alpha,
    name=args.name,
    label_smoothing=args.ls,
    T=args.T,
)

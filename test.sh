mkdir -p logs
nohup python test_cifar10.py --method test  --gpu 0 --ifAA 1 > logs/cifar10.out 2>&1 &
nohup python test_cifar100.py --method test  --gpu 1 --ifAA 1 > logs/cifar100.out 2>&1 &
# nohup python test_tiny_imagenet.py --method topc-2-1  --gpu 0 --ifAA 1 > logs/tiny_imagenet.out 2>&1 &
# nohup python test_tiny_imagenet.py --method topc-2-2  --gpu 1 --ifAA 1 > logs/tiny_imagenet.out 2>&1 &

# nohup python test_tiny_imagenet.py --method topc-1-1  --gpu 0 --ifAA 0 > logs/tiny_imagenet.out 2>&1 &
# nohup python test_tiny_imagenet.py --method topc-1-1 --gpu 1 --ifAA 1 > logs/tiny_imagenet.out 2>&1 &

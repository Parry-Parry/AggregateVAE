from fire import Fire 
import os

def main():
    os.system('python AggregateVAE/scripts/training/k_grid_eval.py --script AggregateVAE/scripts/training/std/image.py --dataset mnist --datastore ds/dump --outstore train/image/aggr --trainstore ds/mnist --batch_size 16 --gpus 1')
    os.system('python AggregateVAE/scripts/training/k_grid_eval.py --script AggregateVAE/scripts/training/std/image.py --dataset cifar10 --datastore ds/dump --outstore train/image/aggr --trainstore ds/cifar10 --batch_size 16 --gpus 1')
    os.system('python AggregateVAE/scripts/training/k_grid_eval.py --script AggregateVAE/scripts/training/std/image.py --dataset mnist --datastore ds/dump --outstore train/image/aggr --trainstore ds/mnist --batch_size 16 --gpus 1 --vae')
    os.system('python AggregateVAE/scripts/training/k_grid_eval.py --script AggregateVAE/scripts/training/std/image.py --dataset cifar10 --datastore ds/dump --outstore train/image/aggr --trainstore ds/cifar10 --batch_size 16 --gpus 1 --vae')
    os.system('python AggregateVAE/scripts/training/recons_grid_eval.py --script AggregateVAE/scripts/training/recons/image.py --dataset mnist --datastore ds/dump --outstore train/image/aggr --trainstore ds/mnist --batch_size 16 --gpus 1')
    os.system('python AggregateVAE/scripts/training/recons_grid_eval.py --script AggregateVAE/scripts/training/recons/image.py --dataset cifar10 --datastore ds/dump --outstore train/image/aggr --trainstore ds/cifar10 --batch_size 16 --gpus 1')
if __name__ == '__main__':
    Fire(main)
from fire import Fire 
import os

def main():
    os.system('python AggregateVAE/scripts/training/recons_grid_eval_tabular.py --script AggregateVAE/scripts/training/recons/tabular.py --dataset bankjob --datastore ds/bankjob --outstore train/bank/aggr --batch_size 8 --gpus 1')
    os.system('python AggregateVAE/scripts/training/recons_grid_eval_tabular.py --script AggregateVAE/scripts/training/recons/tabular.py --dataset bankbalance --datastore ds/bankbalance --outstore train/bank/aggr --batch_size 8 --gpus 1')
if __name__ == '__main__':
    Fire(main)
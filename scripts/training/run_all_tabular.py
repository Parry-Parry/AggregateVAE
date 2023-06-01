from fire import Fire 
import os

def main():
    os.system('python AggregateVAE/scripts/training/std_grid_eval.py --script AggregateVAE/scripts/training/std/tabular.py --dataset bankbalance --datastore ds/bank/balance --outstore train/bank/aggr --batch_size 8 --gpus 1')
    os.system('python AggregateVAE/scripts/training/std_grid_eval.py --script AggregateVAE/scripts/training/std/tabular.py --dataset bankbalance --datastore ds/bank/balance --outstore train/bank/aggr --batch_size 8 --gpus 1 --vae')
    os.system('python AggregateVAE/scripts/training/recons_grid_eval_tabular.py --script AggregateVAE/scripts/training/recons/tabular.py --dataset bankbalance --datastore ds/bank/balance --outstore train/bank/ --batch_size 8 --gpus 1')
    os.system('python AggregateVAE/scripts/training/std_grid_eval.py --script AggregateVAE/scripts/training/std/tabular.py --dataset bankbalancejob --datastore ds/bank/balancejob --outstore train/bank/aggr --batch_size 8 --gpus 1')
    os.system('python AggregateVAE/scripts/training/std_grid_eval.py --script AggregateVAE/scripts/training/std/tabular.py --dataset bankbalancejob --datastore ds/bank/balancejob --outstore train/bank/aggr --batch_size 8 --gpus 1 --vae')
    os.system('python AggregateVAE/scripts/training/recons_grid_eval_tabular.py --script AggregateVAE/scripts/training/recons/tabular.py --datasetjob bankbalancejob --datastore ds/bank/balancejob --outstore train/bank/aggr --batch_size 8 --gpus 1')
    os.system('python AggregateVAE/scripts/training/std_grid_eval.py --script AggregateVAE/scripts/training/std/tabular.py --dataset bankjob --datastore ds/bank/job --outstore train/bank/aggr --batch_size 8 --gpus 1')
    os.system('python AggregateVAE/scripts/training/std_grid_eval.py --script AggregateVAE/scripts/training/std/tabular.py --dataset bankjob --datastore ds/bank/job --outstore train/bank/aggr --batch_size 8 --gpus 1 --vae')
    os.system('python AggregateVAE/scripts/training/recons_grid_eval_tabular.py --script AggregateVAE/scripts/training/recons/tabular.py --datasetjob bankjob --datastore ds/bank/job --outstore train/bank/aggr --batch_size 8 --gpus 1')
    os.system('python AggregateVAE/scripts/training/std_grid_eval.py --script AggregateVAE/scripts/training/std/tabular.py --dataset bank --datastore ds/bank/std --outstore train/bank/baseline --batch_size 8 --gpus 1')
if __name__ == '__main__':
    Fire(main)
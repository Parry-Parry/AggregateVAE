import argparse
import logging
import math as m
import wandb
import tensorflow as tf

from ClassifierVAE.utils import init_loss, init_temp_anneal, retrieve_dataset, build_dataset
from ClassifierVAE.structures import *
from ClassifierVAE.models.gumbel import multihead_gumbel
from ClassifierVAE.models.layers import init_decoder, init_encoder, init_head
from ClassifierVAE.models.internal_layers import init_convnet, init_convtransposenet, init_densenet
from ClassifierVAE.training.wrapper import wrapper

tfk = tf.keras
tfkl = tfk.layers

### CONSTANTS ### 
ENCODER_STACK = [32, 64, 128]
DECODER_STACK = [64, 32, 32]
HEAD_INTERMEDIATE = [64, 128]
HEAD_STACK = [256, 128]

ACTIVATION = 'relu'

INIT_TAU = 1.0
ANNEAL_RATE = 1e-3
MIN_TAU = 0.1

HARD = False
N_DIST = 20

### ARGS ### 

parser = argparse.ArgumentParser(description='Training of ensemble classifier with epsilon reconstruction')

parser.add_argument('-project', type=str, default=None, help='Wandb Project name')
parser.add_argument('-uname', type=str, default=None, help='Wandb username')
parser.add_argument('-dataset', type=str, default=None, help='Training Dataset, Supported: CIFAR10 & CIFAR100, MNIST')
parser.add_argument('-batch', type=int, default=16, help='Batch size')
parser.add_argument('-k', type=int, help='How much aggregation to perform upon the dataset')
parser.add_argument('-epochs', type=int, default=15, help='Number of epochs to train')
parser.add_argument('-heads', type=int, default=3, help='Number of generators')

parser.add_argument('--partition_path', type=str, help='Where to retrieve and save aggregate data')
parser.add_argument('--seed', type=int, default=8008, help='Seed for random generator')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate, Default 0.0001')


def main(args):
    ### COLLECT ARGS & INIT LOGS ###

    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    NUM_HEADS = args.heads
    EPOCHS = args.epochs
    MULTIHEAD = NUM_HEADS > 1

    tau = tf.Variable(INIT_TAU, trainable=False)

    config = {
        'learning_rate' : args.lr,
        'batch_size' : args.batch,
        'K' : args.k,
        'p' : NUM_HEADS, 
        'num distributions' : N_DIST,
        'dataset' : args.dataset,
        'initial temperature' : INIT_TAU,
        'temperature anneal rate' : ANNEAL_RATE,
        'minimum tau' : MIN_TAU
    }

    wandb.init(project=args.project, entity=args.uname, config=config)
    config = wandb.config

    ### BUILD DATASET ###

    logger.info(f'Building Dataset {config.dataset} with clusters {config.K}')

    name, data = retrieve_dataset(args.dataset, None) # Retreive true dataset
    x_train, x_test, y_train, y_test = data
    dataset = Dataset(name, x_train, x_test, y_train, y_test)

    train_set, test_set, N_CLASS = build_dataset(dataset, config.K, args.partition_path, config.batch_size, args.seed)

    out_dim = tuple(dataset.x_train.shape[1:]) + (1,)

    ### INITIALIZE CONFIGS ###

    latent_square = m.floor(m.sqrt(N_CLASS * N_DIST))

    encoder_internal = init_convnet(ENCODER_STACK, dropout_rate=0.25, pooling=False, flatten=True)
    decoder_internal = init_convtransposenet(DECODER_STACK, kernel_size=3, dropout_rate=None, flatten=True)
    head_intermediate = init_convnet(HEAD_INTERMEDIATE, dropout_rate=0.25, flatten=True)
    head_internal = init_densenet(HEAD_STACK, dropout_rate=0.25)


    encoder_config = Encoder_Config(N_CLASS, N_DIST, encoder_internal, ACTIVATION, tau)
    decoder_config = Decoder_Config(N_CLASS, N_DIST, decoder_internal, ACTIVATION, latent_square, out_dim, tau)
    head_config = Head_Config(N_CLASS, head_intermediate, head_internal, ACTIVATION, out_dim)

    ### INITIALIZE MODEL ###

    encoder_func = init_encoder(encoder_config)
    decoder_func = init_decoder(decoder_config)
    head_func = init_head(head_config)

    model_config = Model_Config(NUM_HEADS, encoder_func, decoder_func, head_func, None, N_CLASS, out_dim, HARD)

    model = multihead_gumbel(model_config)

    ### INITIALIZE WRAPPER ###
    loss = init_loss(MULTIHEAD, N_DIST, N_CLASS)
    optim = tfk.optimizers.Adam(learning_rate=args.lr)

    temp_anneal = init_temp_anneal(INIT_TAU, MIN_TAU, ANNEAL_RATE)
    acc_metric = tfk.metrics.CategoricalAccuracy

    wrapper_config = Wrapper_Config(model, loss, optim, EPOCHS, temp_anneal, acc_metric)

    train_wrapper = wrapper(wrapper_config)

    ### TRAINING ### 

    logger.info(f'Training model with {config.p} heads for {args.epochs} epochs...')

    train_wrapper.fit(train_set, test_set, wandb)

    return 0



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)    
    code = main(parser.parse_args())
    if code == 0:
        print("Executed Successfully")
    else:
        print("Error see logs")
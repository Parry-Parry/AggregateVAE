from ClassifierVAE.models.internal_layers import init_convnet
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
ENCODER_STACK = [64, 128, 256]
DECODER_STACK = [256, 128, 64, 64]
HEAD_STACK = [256, 128]

ACTIVATION = 'relu'

INIT_TAU = 1.0
ANNEAL_RATE = 1e-3
MIN_TAU = 0.1

HARD = False
N_DIST = 20


def main(args):
    ### COLLECT ARGS & INIT LOGS ###
    NUM_HEADS = args.heads
    EPOCHS = args.epochs
    MULTIHEAD = NUM_HEADS > 1

    tau = tf.Variable(INIT_TAU, trainable=False)

    INTERMEDIATE = None

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

    name, data = retrieve_dataset(args.dataset, None) # Retreive true dataset
    x_train, x_test, y_train, y_test = data
    dataset = Dataset(name, x_train, x_test, y_train, y_test)

    train_set, test_set, N_CLASS = build_dataset(dataset, config.K, args.partition_path, config.batch_size, args.seed, config.p, config.epsilon)

    train_set = None 
    test_set = None

    ### INITIALIZE CONFIGS ###

    encoder_internal = init_convnet(ENCODER_STACK, dropout_rate=0.25, flatten=True)
    decoder_internal = init_convtransposenet(DECODER_STACK, dropout_rate=0.25, flatten=True)
    head_internal = init_convnet(HEAD_STACK, dropout_rate=0.25, flatten=True)


    encoder_config = Encoder_Config(N_CLASS, N_DIST, encoder_internal, ACTIVATION, tau)
    decoder_config = Decoder_Config(N_CLASS, N_DIST, decoder_internal, ACTIVATION, tau)
    head_config = Head_Config(N_CLASS, INTERMEDIATE, head_internal, ACTIVATION)

    ### INITIALIZE MODEL ###

    encoder_func = init_encoder(encoder_config)
    decoder_func = init_decoder(decoder_config)
    head_func = init_head(head_config)

    input_layer = tfkl.Input(shape=dataset.x_train.shape[1:])

    model_config = Model_Config(NUM_HEADS, encoder_func, decoder_func, head_func, input_layer, N_CLASS, HARD)

    model = multihead_gumbel(model_config)

    ### INITIALIZE WRAPPER ###
    loss = init_loss(MULTIHEAD)
    optim = tfk.optimizers.Adam(learning_rate=args.lr)

    temp_anneal = init_temp_anneal(INIT_TAU, MIN_TAU, ANNEAL_RATE)
    acc_metric = tfk.metrics.CategoricalAccuracy

    wrapper_config = Wrapper_Config(model, loss, optim, EPOCHS, temp_anneal, acc_metric)

    train_wrapper = wrapper(wrapper_config)

    ### TRAINING ### 
    train_wrapper.fit(train_set, test_set, wandb)

    return 0



if __name__ == '__main__':
    err = main()
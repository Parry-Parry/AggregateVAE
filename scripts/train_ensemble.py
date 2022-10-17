import wandb
import tensorflow as tf

from ClassifierVAE.utils import Decoder_Config, Encoder_Config, Head_Config, Model_Config, Wrapper_Config, init_loss, init_temp_anneal
from ClassifierVAE.models.gumbel import multihead_gumbel
from ClassifierVAE.models.layers import init_decoder, init_encoder, init_head
from ClassifierVAE.training.wrapper import wrapper

tfk = tf.keras

### CONSTANTS ### 
BATCH = 32 
ENCODER_STACK = [512, 256]
DECODER_STACK = [256, 512]
HEAD_STACK = [256, 128]

ACTIVATION = 'relu'

INIT_TAU = 1.0
ANNEAL_RATE = 1e-3
MIN_TAU = 0.1

HARD = False 
N_DIST = 20


def main(args):
    ### COLLECT ARGS ###
    NUM_HEADS = args.heads
    EPOCHS = args.epochs
    MULTIHEAD = NUM_HEADS > 1
    N_CLASS = None

    tau = tf.Variable(INIT_TAU, trainable=False)

    INTERMEDIATE = None

    ### BUILD DATASET ###

    train_set = None 
    test_set = None

    ### INITIALIZE CONFIGS & MODEL ###

    config = {
        'learning_rate' : args.lr,
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

    encoder_config = Encoder_Config(N_CLASS, N_DIST, ENCODER_STACK, ACTIVATION, tau)
    decoder_config = Decoder_Config(N_CLASS, N_DIST, DECODER_STACK, ACTIVATION, tau)
    head_config = Head_Config(N_CLASS, INTERMEDIATE, HEAD_STACK, ACTIVATION)

    encoder_func = init_encoder(encoder_config)
    decoder_func = init_decoder(decoder_config)
    head_func = init_head(head_config)

    model_config = Model_Config(NUM_HEADS, encoder_func, decoder_func, head_func, N_CLASS, HARD)

    model = multihead_gumbel(model_config)

    ### INIT WRAPPER ###
    loss = init_loss(MULTIHEAD)
    optim = tfk.optimizers.Adam(learning_rate=args.lr)

    temp_anneal = init_temp_anneal(INIT_TAU)
    acc_metric = tfk.metrics.CategoricalAccuracy

    wrapper_config = Wrapper_Config(model, loss, optim, EPOCHS, temp_anneal, acc_metric)

    train_wrapper = wrapper(wrapper_config)

    ### TRAINING ### 
    train_wrapper.fit(train_set, test_set, wandb)

    return 0



if __name__ == '__main__':
    err = main()
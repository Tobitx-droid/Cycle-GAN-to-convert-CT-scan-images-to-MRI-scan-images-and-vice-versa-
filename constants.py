"""
    This file contains all the constants used in the project.
"""

DATASET_ROOT_DIR = 'dataset/images'
SAMPLES_DIR = 'dataset/samples'
TRAIN_A = 'trainA'
TRAIN_B = 'trainB'
TEST_A = 'testA'
TEST_B = 'testB'
UNSEEN_DEMO_IMAGES = 'dataset/unseen_demo_images'
UNSEEN_CT = 'ct'
UNSEEN_MRI = 'mri'
DATASET_TRAIN_MODE = 'train'
DATASET_TEST_MODE = 'test'
SHUFFLE = True
EPOCHS = 0
N_EPOCHS = 10
BATCH_SIZE = 4
VAL_BATCH_SIZE = 16
LEARNING_RATE = 0.0002
DECAY_START_EPOCH = 100
BETA_1 = 0.5
BETA_2 = 0.999
N_CPU = 2  # change to a number based on the device for the training (8)
IMG_SIZE = 128
CHANNELS = 3
N_CRITIC = 5
SAMPLE_INTERVAL = 100
NUM_RESIDUAL_BLOCKS = 19
LAMBDA_CYC = 10.0
LAMBDA_ID = 5.0
NUM_WORKERS = 1

ACTIVATION_FUNCTION = None

# Cycle Consistent GAN constants
KERNEL_SIZE = 4
STRIDE = 2
PADDING = 1
OUT_CHANNELS = 1
IN_CHANNELS = 512

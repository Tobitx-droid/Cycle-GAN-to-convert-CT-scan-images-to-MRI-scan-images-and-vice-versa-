"""
This file contains the hyperparameters for the model.
"""


##############################################
# Defining all hyperparameters
##############################################
class Hyperparameters(object):
    """
        This class contains the hyperparameters for the model.
    """

    def __init__(self, **kwargs) -> None:
        self.epoch = None,
        self.n_epochs = None,
        self.dataset_train_mode = None,
        self.dataset_test_mode = None,
        self.batch_size = None,
        self.lr = None,
        self.decay_start_epoch = None,
        self.b1 = None,
        self.b2 = None,
        self.n_cpu = None,
        self.img_size = None,
        self.channels = None,
        self.n_critic = None,
        self.sample_interval = None,
        self.num_residual_blocks = None,
        self.lambda_cyc = None,
        self.lambda_id = None,
        self.__dict__.update(kwargs)


###############################################
# create an instance of Hyperparameters
###############################################

hp = Hyperparameters(
    epoch=0,
    n_epochs=200,
    dataset_train_mode="train",
    dataset_test_mode="test",
    batch_size=4,
    lr=0.0002,
    decay_start_epoch=100,
    b1=0.5,
    b2=0.999,
    n_cpu=8,
    img_size=128,
    channels=3,
    n_critic=5,
    sample_interval=100,
    num_residual_blocks=19,
    lambda_cyc=10.0,
    lambda_id=5.0,
)

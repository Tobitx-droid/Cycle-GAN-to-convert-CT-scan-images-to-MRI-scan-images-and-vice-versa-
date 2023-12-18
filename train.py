import itertools
import time
import datetime
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from model.data_loader import ImageDataset
from IPython.display import clear_output

from model.utils import *
from model.cycle_gan import *

"""
    So generally both torch.Tensor and torch.cuda.Tensor are equivalent.
    You can do everything you like with them both.
    The key difference is just that torch.
    Tensor occupies CPU memory while torch.cuda.Tensor occupies GPU memory.
    Of course operations on a CPU Tensor are computed with
    while operations for the GPU / CUDA Tensor are computed on GPU.
"""
cuda = True if torch.cuda.is_available() else False
print("Using CUDA" if cuda else "Not using CUDA")

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

##############################################
# Defining Image Transforms to apply
##############################################

transforms_ = [
    transforms.Resize(
        (hp.img_size, hp.img_size),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

train_dataloader = DataLoader(
    ImageDataset(
        DATASET_ROOT_DIR,
        mode=DATASET_TRAIN_MODE,
        transforms_=transforms_
    ),
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS,
)
val_dataloader = DataLoader(
    ImageDataset(
        DATASET_ROOT_DIR,
        mode=DATASET_TEST_MODE,
        transforms_=transforms_
    ),
    batch_size=VAL_BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS,
)

# data loader for unseen images


##############################################
# SAMPLING IMAGES
##############################################


def save_img_samples(batches_done):
    """Saves a generated sample from the test set"""
    print("batches_done ", batches_done)
    imgs = next(iter(val_dataloader))

    Gen_AB.eval()
    Gen_BA.eval()

    real_a = Variable(imgs["A"].type(Tensor))
    fake_b = Gen_AB(real_a)
    real_b = Variable(imgs["B"].type(Tensor))
    fake_a = Gen_BA(real_b)
    # Arrange images along x-axis
    real_a = make_grid(real_a, nrow=16, normalize=True)
    real_b = make_grid(real_b, nrow=16, normalize=True)
    fake_a = make_grid(fake_a, nrow=16, normalize=True)
    fake_b = make_grid(fake_b, nrow=16, normalize=True)
    # Arrange images along y-axis
    image_grid = torch.cat((real_a, fake_b, real_b, fake_a), 1)

    path = SAMPLES_DIR + "/%s.png" % batches_done

    save_image(image_grid, path, normalize=False)
    return path


##############################################
# SETUP, LOSS, INITIALIZE MODELS and BUFFERS
##############################################

# Creating criterion object (Loss Function) that will
# measure the error between the prediction and the target.
criterion_GAN = torch.nn.MSELoss()

criterion_cycle = torch.nn.L1Loss()

criterion_identity = torch.nn.L1Loss()

input_shape = (hp.channels, hp.img_size, hp.img_size)

##############################################
# Initialize generator and discriminator
##############################################

Gen_AB = GeneratorResNet(input_shape, hp.num_residual_blocks)
Gen_BA = GeneratorResNet(input_shape, hp.num_residual_blocks)

Disc_A = Discriminator(input_shape)
Disc_B = Discriminator(input_shape)

if cuda:
    Gen_AB = Gen_AB.cuda()
    Gen_BA = Gen_BA.cuda()
    Disc_A = Disc_A.cuda()
    Disc_B = Disc_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

##############################################
# Initialize weights
##############################################

Gen_AB.apply(initialize_conv_weights_normal)
Gen_BA.apply(initialize_conv_weights_normal)

Disc_A.apply(initialize_conv_weights_normal)
Disc_B.apply(initialize_conv_weights_normal)

##############################################
# Buffers of previously generated samples
##############################################

fake_a_buffer = ReplayBuffer()

fake_b_buffer = ReplayBuffer()

##############################################
# Defining all Optimizers
##############################################
optimizer_G = torch.optim.Adam(
    itertools.chain(Gen_AB.parameters(), Gen_BA.parameters()),
    lr=hp.lr,
    betas=(hp.b1, hp.b2),
)
optimizer_Disc_A = torch.optim.Adam(
    Disc_A.parameters(),
    lr=hp.lr,
    betas=(hp.b1, hp.b2)
)

optimizer_Disc_B = torch.optim.Adam(
    Disc_B.parameters(),
    lr=hp.lr,
    betas=(hp.b1, hp.b2)
)

##############################################
# Learning rate update schedulers
##############################################
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step
)

lr_scheduler_Disc_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_Disc_A,
    lr_lambda=LambdaLR(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step,
)

lr_scheduler_Disc_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_Disc_B,
    lr_lambda=LambdaLR(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step,
)


##############################################
# Final Training Function
##############################################

def train(
        Gen_BA,
        Gen_AB,
        Disc_A,
        Disc_B,
        train_dataloader,
        n_epochs,
        criterion_identity,
        criterion_cycle,
        lambda_cyc,
        criterion_GAN,
        optimizer_G,
        fake_a_buffer,
        fake_b_buffer,
        clear_output,
        optimizer_Disc_A,
        optimizer_Disc_B,
        Tensor,
        sample_interval,
        lambda_id,
) -> None:
    # TRAINING
    prev_time = time.time()
    for epoch in range(hp.epoch, n_epochs):
        for i, batch in enumerate(train_dataloader):

            # Set model input
            real_a = Variable(batch["A"].type(Tensor))
            real_b = Variable(batch["B"].type(Tensor))

            # Adversarial ground truths i.e. target vectors
            # 1 for real images and 0 for fake generated images
            valid = Variable(
                Tensor(np.ones((real_a.size(0), *Disc_A.output_shape))),
                requires_grad=False,
            )

            fake = Variable(
                Tensor(np.zeros((real_a.size(0), *Disc_A.output_shape))),
                requires_grad=False,
            )

            ###################################
            # # Train Generators A->B and B->A
            ###################################

            Gen_AB.train()
            Gen_BA.train()

            """
            PyTorch stores gradients in a mutable data structure. 
            So we need to set it to a clean state before we use it.
            Otherwise, it will have old information from a previous iteration.
            """
            optimizer_G.zero_grad()

            # Identity loss
            # First pass real_a images to the Generator, that will generate A-domains images
            loss_id_A = criterion_identity(Gen_BA(real_a), real_a)

            # Then pass real_b images to the Generator, that will generate B-domains images
            loss_id_B = criterion_identity(Gen_AB(real_b), real_b)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN losses for GAN_AB
            fake_b = Gen_AB(real_a)

            loss_GAN_AB = criterion_GAN(Disc_B(fake_b), valid)

            # GAN losses for GAN_BA
            fake_a = Gen_BA(real_b)

            loss_GAN_BA = criterion_GAN(Disc_A(fake_a), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle Consistency losses
            reconstructed_A = Gen_BA(fake_b)

            """
            Forward Cycle Consistency Loss
            Forward cycle loss:  lambda * ||G_BtoA(G_AtoB(A)) - A|| (Equation 2 in the paper)
            Compute the cycle consistency loss by comparing the reconstructed_A 
            images with real real_a  images of domain A.
            Lambda for cycle loss is 10.0. Penalizing 10 times and forcing to learn the translation.
            """
            loss_cycle_A = criterion_cycle(reconstructed_A, real_a)

            """
            Backward Cycle Consistency Loss
            Backward cycle loss: lambda * ||G_AtoB(G_BtoA(B)) - B|| (Equation 2 of the Paper)
            Compute the cycle consistency loss by comparing the reconstructed_B 
            images with real real_b images of domain B.
            Lambda for cycle loss is 10.0. Penalizing 10 times and forcing to learn the translation.
            """
            reconstructed_B = Gen_AB(fake_a)

            loss_cycle_B = criterion_cycle(reconstructed_B, real_b)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            """
            Finally, Total Generators Loss and Back propagation
            Add up all the Generators loss and cyclic loss (Equation 3 of paper.
            Also Equation I the code representation of the equation) and perform backpropagation with optimization.
            """
            loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

            loss_G.backward()

            """
            Now we just need to update all the parameters!
            Θ_{k+1} = Θ_k - η * ∇_Θ ℓ(y_hat, y)
            """
            optimizer_G.step()

            #########################
            #  Train Discriminator A
            #########################

            optimizer_Disc_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(Disc_A(real_a), valid)
            # Fake loss (on batch of previously generated samples)

            fake_a_ = fake_a_buffer.push_and_pop(fake_a)

            loss_fake = criterion_GAN(Disc_A(fake_a_.detach()), fake)

            """ Total loss for Disc_A
            And I divide by 2 because as per Paper - "we divide the objective by 2 while
            optimizing D, which slows down the rate at which D learns,
            relative to the rate of G."
            """
            loss_Disc_A = (loss_real + loss_fake) / 2

            """ do backpropagation i.e.
            ∇_Θ will get computed by this call below to backward() """
            loss_Disc_A.backward()

            """
            Now we just need to update all the parameters!
            Θ_{k+1} = Θ_k - η * ∇_Θ ℓ(y_hat, y)
            """
            optimizer_Disc_A.step()

            #########################
            #  Train Discriminator B
            #########################

            optimizer_Disc_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(Disc_B(real_b), valid)

            # Fake loss (on batch of previously generated samples)
            fake_b_ = fake_b_buffer.push_and_pop(fake_b)

            loss_fake = criterion_GAN(Disc_B(fake_b_.detach()), fake)

            """ Total loss for Disc_B
            And I divide by 2 because as per Paper - "we divide the objective by 2 while
            optimizing D, which slows down the rate at which D learns,
            relative to the rate of G."
            """
            loss_Disc_B = (loss_real + loss_fake) / 2

            """ do backpropagation i.e.
            ∇_Θ will get computed by this call below to backward() """
            loss_Disc_B.backward()

            """
            Now we just need to update all the parameters!
            Θ_{k+1} = Θ_k − η * ∇_Θ ℓ(y_hat, y)
            """
            optimizer_Disc_B.step()

            loss_D = (loss_Disc_A + loss_Disc_B) / 2

            ##################
            #  Log Progress
            ##################

            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + i

            batches_left = n_epochs * len(train_dataloader) - batches_done

            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time)
            )
            prev_time = time.time()

            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(train_dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % sample_interval == 0:
                clear_output()
                plot_output(save_img_samples(batches_done), 30, 40)

    # Save model checkpoints
    torch.save(Gen_AB.state_dict(), "../trained/Gen_AB.pth")
    torch.save(Gen_BA.state_dict(), "../trained/Gen_BA.pth")
    torch.save(Disc_A.state_dict(), "../trained/Disc_A.pth")
    torch.save(Disc_B.state_dict(), "../trained/Disc_B.pth")


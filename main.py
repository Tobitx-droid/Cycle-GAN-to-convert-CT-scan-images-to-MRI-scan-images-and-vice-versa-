"""
    Main program for Cycle Consistent GAN
"""

# # # imports
from model.train import *
from utils.distillation import distillation_training


# # #
# # # concept diagram
# # # [Input] ---> CT Network ---> [Distillation Loss] --->
# # # MRI Network ---> [Generator Loss] ---> [Output]
# # #

def main():
    """
        Main program starts here pulling resources from other files
    """

    ##############################################
    # Execute the Final Training Function
    ##############################################

    train(
        Gen_BA=Gen_BA,
        Gen_AB=Gen_AB,
        Disc_A=Disc_A,
        Disc_B=Disc_B,
        train_dataloader=train_dataloader,
        n_epochs=hp.n_epochs,
        criterion_identity=criterion_identity,
        criterion_cycle=criterion_cycle,
        lambda_cyc=hp.lambda_cyc,
        criterion_GAN=criterion_GAN,
        optimizer_G=optimizer_G,
        fake_a_buffer=fake_a_buffer,
        fake_b_buffer=fake_b_buffer,
        clear_output=clear_output,
        optimizer_Disc_A=optimizer_Disc_A,
        optimizer_Disc_B=optimizer_Disc_B,
        Tensor=Tensor,
        sample_interval=hp.sample_interval,
        lambda_id=hp.lambda_id,
    )

    # ---------------------------------------------------------------------------------------
    # use the distillation network from the utils folder to train the distillation network
    # pass in the generator networks and the dataloader for the testing data
    # ---------------------------------------------------------------------------------------

    distillation_training(
        teacher_model_path="trained/Gen_BA.pth",
        student_model_path="trained/Gen_AB.pth",
        similarities_path="trained/similarities.npy",
        x_test=val_dataloader.dataset,
        y_test=val_dataloader.dataset,
        epoch=100
    )


if __name__ == "__main__":
    main()

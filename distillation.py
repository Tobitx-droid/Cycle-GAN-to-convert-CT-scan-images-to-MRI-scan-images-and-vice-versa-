import torch
import torch.nn as nn
import numpy as np
from .srpkd import SRPKD


# Define the distillation loss function
def distillation_loss(student_outputs, teacher_outputs):
    # You can use MSE loss or any other suitable loss function here
    # The loss function should be differentiable
    return nn.MSELoss()(student_outputs, teacher_outputs)


def calculate_similarities(teacher_feature_maps, student_feature_maps):
    # Calculate the pairwise similarities between all feature pixels in the two models.
    similarities = np.matmul(teacher_feature_maps, student_feature_maps.T)

    # Return the similarities.
    return similarities


def distillation_training(
        x_test: any,
        y_test: any,
        teacher_model_path: str,
        student_model_path: str,
        similarities_path: str,
        epoch: int = 100,
) -> None:
    """
        Utilizing the SRPKD approach, train the student model to mimic the teacher model.
    """

    # Load the teacher and student models.
    teacher_model = torch.load(teacher_model_path)
    student_model = torch.load(student_model_path)

    # Extract the feature maps from the teacher and student models.
    teacher_feature_maps = teacher_model.predict(x_test)
    student_feature_maps = student_model.predict(x_test)

    # Calculate the similarities between the feature pixels in the two models.
    similarities = calculate_similarities(teacher_feature_maps, student_feature_maps)

    # Save the similarities to a file.
    np.save(similarities_path, similarities)

    # Load the similarities between the feature pixels in the teacher and student models.
    similarities = np.load(similarities_path)
    # Convert the similarities to a torch tensor.
    similarities = torch.tensor(similarities)
    # Create the optimizer.
    optimizer = distillation_loss(student_model, teacher_model)

    # Create the SRPKD object.
    srpkd = SRPKD(
        teacher_model=teacher_model,
        student_model=student_model,
        similarities=similarities,
        optimizer=optimizer,
        x_test=x_test,
        y_test=y_test,
        epoch=epoch,
    )
    # Train the student model.
    srpkd.train_distillation()
    # Evaluate the student model.
    accuracy = srpkd.evaluate_distillation()
    print(f"Accuracy: {accuracy}")

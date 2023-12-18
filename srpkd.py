"""
    The SRPKD approach has been shown to be effective in improving the performance of student CycleGAN generators.
"""

import torch
import numpy as np


class SRPKD:
    def __init__(
            self,
            teacher_model: any,
            student_model: any,
            similarities: any,
            x_test: any,
            y_test: any,
            optimizer: any,
            atol: any = 1e-5,
            epoch: int = 100,
    ) -> None:
        self.optimizer = optimizer
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.similarities = similarities
        self.x_test = x_test
        self.y_test = y_test
        self.atol = atol
        self.epochs = epoch

    def train_distillation(self) -> None:
        """
            Utilizing the SRPKD approach, train the student model to mimic the teacher model.
        """
        for epoch in range(self.epochs):
            # Calculate the predicted similarities for the student model.
            predicted_similarities = self.student_model(self.teacher_model(self.x_test))

            # Calculate the loss between the predicted similarities and the ground truth similarities.
            loss = torch.mean(torch.pow(predicted_similarities - self.similarities, 2))

            # Back-propagate the loss and update the student model's parameters.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print the loss every epoch.
            print(f"Epoch: {epoch} Loss: {loss}")

    def evaluate_distillation(self) -> any:  # float | any
        """
            Evaluate the student model.
        """
        # Calculate the predicted similarities for the student model.
        predicted_similarities = self.student_model(self.teacher_model(self.x_test))

        # Calculate the accuracy of the student model.
        accuracy = np.mean(np.isclose(predicted_similarities, self.similarities, atol=self.atol))

        return accuracy

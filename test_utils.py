"""
    Unit tests for the model.utils module.
"""

from PIL import Image
import numpy as np

from model.utils import convert_to_rgb

########################################################
# Unit tests for convert_to_rgb function
########################################################


def test_convert_to_rgb():
    """
        Unit test for convert_to_rgb function
    """
    # Create a sample image with a single channel (grayscale)
    image = Image.fromarray(np.zeros((100, 100), dtype=np.uint8), mode='L')

    # Convert the image to RGB format
    rgb_image = convert_to_rgb(image)

    # Check the output type and mode
    assert isinstance(rgb_image, Image.Image)
    assert rgb_image.mode == "RGB"

    # Check the output size
    assert rgb_image.size == image.size

    print("Unit test passed!")

# Run the unit test
# test_convert_to_rgb()

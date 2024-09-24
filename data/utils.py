import numpy as np
import tensorflow as tf
from scipy import ndimage
from sklearn.metrics import confusion_matrix
from tensorflow import keras


def get_confusion_matrix(model: keras.Model, test_data: tf.data.Dataset, labels: np.ndarray, 
                         class_names: list, normalize: str = 'true') -> np.ndarray:
    """Computes the sklearn confusion matrix of the model.

    Args:
        model (keras.model): The model to evaluate.
        test_data (tf.data.Dataset): The test dataset.
        labels (np.ndarray): The labels of the test dataset.
        class_names (list): A list of all class names.
        normalize (bool): Normalizes confusion matrix over the true (rows), predicted (columns) 
            conditions or all the population. If None, confusion matrix will not be normalized.
            Defaults to 'true'

    Returns:
        np.ndarray: The sklearn confusion matrix as a np.ndarray.
    """
    out = model.predict(test_data)
    preds = np.argmax(out, axis=1)
    return confusion_matrix(labels, preds, labels=np.arange(len(class_names)), normalize=normalize)


def rotate_image(img: np.ndarray, rot: float) -> np.ndarray:
    """Rotates an image by the given amount. If the image is one-dimensional,
    it is interpreted as a square image.

    Args:
        img (np.ndarray): The image to rotate.
        rot (float): The degree of rotation between 0 and 360.

    Returns:
        np.ndarray: The rotated image in the same shape as the input.
    """
    flat = len(img.shape) == 1
    if flat:
        size = int(np.round(np.sqrt(img.shape[0])))
        img = img.reshape((size, size))
        
    img_rot = ndimage.rotate(img, rot, reshape=False)
    
    if flat:
        img_rot = img_rot.reshape(-1)
    return img_rot

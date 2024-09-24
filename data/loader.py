import os
from zipfile import ZipFile

import numpy as np
import requests

data_dir = os.path.realpath(os.path.dirname(__file__))
MNIST_ROT_DIR = os.path.join(data_dir, 'mnist_rot')

MNIST_ROT_URL = "https://www.dropbox.com/s/0fxwai3h84dczh0/mnist_rotation_new.zip?dl=1"


def download_mnist_rot():
    """Downloads the MNIST-rot dataset into data/mnist_rot, seperated into three files:
        - rotated_train.npz
        - rotated_valid.npz
        - rotated_test.npz
    """
    
    if not os.path.exists(MNIST_ROT_DIR):
        os.makedirs(MNIST_ROT_DIR)
    
    zip_path = os.path.join(MNIST_ROT_DIR, 'mnist_rot.zip')
        
    print('Downloading the MNIST-rot dataset...')
    with open(zip_path, 'wb') as f:
        f.write(requests.get(MNIST_ROT_URL, stream=True).content)
        
    print('Extracting...')
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(MNIST_ROT_DIR)
        
    # move files
    for file_name in os.listdir(os.path.join(MNIST_ROT_DIR, 'mnist_rotation_new')):
        src = os.path.join(MNIST_ROT_DIR, 'mnist_rotation_new', file_name)
        dest = os.path.join(MNIST_ROT_DIR, file_name)
        os.rename(src, dest)
    os.rmdir(os.path.join(MNIST_ROT_DIR, 'mnist_rotation_new'))
    os.remove(zip_path)

    print('Successfully retrieved the MNIST-rot dataset.')  
          

def mnist_rot_present() -> bool:
    """Return true iff all MNIST-rot datasets are already downloaded."""
    return all(os.path.isfile(os.path.join(MNIST_ROT_DIR, f'rotated_{x}.npz') )
               for x in ('train', 'valid', 'test'))          


def mnist_rot_np() -> tuple[dict]:
    """Loads the MNIST-rot dataset as numpy arrays. The dataset is downloaded if not 
    already done.

    Returns:
        tuple[dict]: The training, validation and test dataset in this order. Each dataset
        is a dict of np.ndarrays with keys 'x', 'y'.
    """
    if not mnist_rot_present():
        download_mnist_rot()
        
    train = np.load(os.path.join(MNIST_ROT_DIR, 'rotated_train.npz'))
    valid = np.load(os.path.join(MNIST_ROT_DIR, 'rotated_valid.npz'))
    test = np.load(os.path.join(MNIST_ROT_DIR, 'rotated_test.npz'))

    return train, valid, test
    
    
def mnist_rot_tf(batch_size: int = 0, shuffle_buffer_size: int = 0, reshufle: bool = True, 
                 drop_remainder: bool = False, normalize: bool = False) -> 'tuple[tf.data.Dataset]':
    """Creates tensorflow datasets of the MNIST-rot dataset. The dataset if downloaded
    if not already done.

    Args:
        batch_size (int, optional): The size of each mini-batch. Set to 0 for no batching. Defaults to 0.
        shuffle_buffer_size (int, optional): The buffer size used for shuffling the training data. 
            Set to 0 to disable shuffling. Defaults to 0.
        reshufle (bool, optional): Whether the dataset should be reshuffled each time it is iterated over. 
        Defaults to True.
        drop_remainder (bool, optional): Whether the last batch should be dropped in the case it has fewer 
            than batch_size elements. Defaults to False.
        normalize (bool, optional): Whether to normalize the data by subtracting mean and dividing by the 
            standard deviation. Defaults to False.

    Returns:
        tuple[tf.data.Dataset]: The training, validation and test datasets in this order.
    """
    import tensorflow as tf
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    train, valid, test = mnist_rot_np()
    
    x_train = train['x']
    y_train = train['y']
    x_valid = valid['x']
    y_valid = valid['y']
    x_test = test['x']
    y_test = test['y']
    
    if normalize:
        mean = np.mean(x_train)
        x_train -= mean
        x_valid -= mean
        x_test -= mean
        std = np.std(x_train)
        x_train /= std
        x_valid /= std
        x_test /= std

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_dataset = train_dataset.cache()
    if shuffle_buffer_size > 0:
        train_dataset = train_dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=reshufle)
    
    if batch_size > 0: 
        train_dataset = train_dataset.batch(batch_size, drop_remainder).prefetch(AUTOTUNE)
        valid_dataset = valid_dataset.batch(batch_size, drop_remainder).prefetch(AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size, drop_remainder).prefetch(AUTOTUNE)
    
    return train_dataset, valid_dataset, test_dataset

if __name__ == '__main__':
    # download dataset if script is executed
    if not mnist_rot_present():
        download_mnist_rot()
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from sklearn.metrics import ConfusionMatrixDisplay
from typing import Iterable


def image_grid(images, labels=None, title='', cols=6, one_row = False, figsize=None, cmap='Blues', ticks=False):
    """Shows the images in a grid.

    Args:
        images (list[np.ndarray]): A list of images to show.
        labels (list[str], optional): Labels of the images. Defaults to None.
        title (str, optional): The title of the figure. Defaults to ''.
        cols (int, optional): The number of columns. Defaults to 6.
        one_row (bool, optional): Whether to show all images in a signle row. This overrides cols.
            Defaults to False.
        figsize (tuple, optional): The size of the figre. Defaults to None.
        cmap (str, optional): The color map used to visualize the image. Defaults to 'Blues'.
        ticks (bool, optional): Whether ticks schould be shown on the axes. Defaults to False.
    """
    if labels is None: labels = ['']*len(images)
    if one_row or cols > len(images): cols = len(images)
        
    rows = int(np.ceil(len(images)/cols))
    if figsize is None: figsize = (5*cols, 5*rows)
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    
    for i, (img, label) in enumerate(zip(images, labels)):
        if len(img.shape) == 1:
            img_size = int(np.round(np.sqrt(np.prod(img.shape))))
            img = np.reshape(img, (img_size, img_size))
        
        ax = plt.subplot(rows, cols, i+1)
        if not ticks:
            plt.xticks([])
            plt.yticks([])
        plt.grid(False)
        plt.imshow(img, cmap=cmap)
        ax.set_title(label)
        
    fig.tight_layout()
    plt.show()


def complex_image_grid(images, labels=None, title='', cols=6, one_row = False, figsize=None, cmap='plasma', ticks=False):
    """Shows the complex images in a grid.

    Args:
        images (list[np.ndarray]): A list of complex images to show.
        labels (list[str], optional): Labels of the images. Defaults to None.
        title (str, optional): The title of the figure. Defaults to ''.
        cols (int, optional): The number of columns. Defaults to 6.
        one_row (bool, optional): Whether to show all images in a signle row. This overrides cols.
            Defaults to False.
        figsize (tuple, optional): The size of the figre. Defaults to None.
        cmap (str, optional): The color map used to visualize the image. Defaults to 'plasma'.
        ticks (bool, optional): Whether ticks schould be shown on the axes. Defaults to False.
    """
    if labels is None: labels = ['']*len(images)
    if one_row or cols > len(images): cols = len(images)
    
    rows = int(np.ceil(len(images)/cols))
    if figsize is None: figsize = (5*cols, 5*rows)
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    
    for i, (img, label) in enumerate(zip(images, labels)):
        img_size = int(np.round(np.sqrt(np.prod(img.shape))))
        img = np.reshape(img, (img_size, img_size))
        
        ax = plt.subplot(rows, cols, i+1)
        if not ticks:
            plt.xticks([])
            plt.yticks([])
        plt.grid(False)
        ax.set_title(label)
        
        real = np.real(img)
        imag = np.imag(img)
        magnitudes = np.absolute(img)
        real_normalized = np.divide(real, magnitudes, out=np.zeros(real.shape, dtype=float), where=magnitudes!=0)
        imag_normalized = np.divide(imag, magnitudes, out=np.zeros(real.shape, dtype=float), where=magnitudes!=0)
        
        size = img.shape
        x,y = np.meshgrid(np.linspace(-1.5,1.5,size[0]), -np.linspace(-1.5,1.5,size[1]))
        x_border, y_border = (2*1.5/size[0])/2, (2*1.5/size[1])/2
        
        # visualize magnitudes
        plt.imshow(magnitudes, 
                   extent=[np.min(x)-x_border,np.max(x)+x_border,np.min(y)-y_border,np.max(y)+y_border], 
                   cmap=cmap, 
                   vmin=0)
        
        # visualize angles
        arrow_length = (2*(1.5+x_border))/size[0]/2
        if size[0] < 4: arrow_length *= 0.8
        plt.quiver(x, y, real_normalized, imag_normalized, angles='xy', scale_units='xy', scale=1/arrow_length)

    fig.tight_layout()
    
    
def confusion_matrix(cm: sklearn.metrics.confusion_matrix, class_names: list):
    """Visualizes a sklearn confusion matrix.

    Args:
        cm (confusion_matrix): The confusion matrix to visualize.
        class_names (list): A list of all class names.
    """
    cm = np.around(cm, 3)
    
    # Show cunfusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    size = len(class_names)
    fig, ax = plt.subplots(figsize=(size,size))
    disp.plot(ax=ax, cmap='Blues', colorbar=False, im_kw={'interpolation':"nearest"})
    plt.tight_layout()
    plt.show()
    
    

def training_history(histories: list['keras.callbacks.History']):
    if not isinstance(histories, Iterable):
        histories = [histories]
    
    x = [range(1, history.epoch[-1]+2) for history in histories]
    train_accuracy, train_loss, val_accuracy, val_loss = [], [], [], []
    for hist in histories:
        train_accuracy.append(hist.history['accuracy'])
        train_loss.append(hist.history['loss'])
        val_accuracy.append(hist.history['val_accuracy'])
        val_loss.append(hist.history['val_loss'])

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for i in range(len(histories)):
        ax[0].plot(x[i], train_accuracy[i], "-", label=f"training accuracy {i+1}")
        ax[0].plot(x[i], val_accuracy[i], "-", label=f"validation accuracy {i+1}")
    ax[0].legend(loc="upper left")
    ax[0].title.set_text('Accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('accuracy')
    # ax[0].set_xticks(x)

    for i in range(len(histories)):
        ax[1].plot(x[i], train_loss[i], "-", label=f"training loss {i+1}")
        ax[1].plot(x[i], val_loss[i], "-", label=f"validation loss {i+1}")
    ax[1].legend(loc="upper left")
    ax[1].title.set_text('Loss')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('loss')
    # ax[1].set_xticks(x)

    plt.tight_layout()
    plt.show()
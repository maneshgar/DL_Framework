from dlLib import dlIO
from tensorflow import keras

def set_params(params, config="custom"):

    if(config == "custom"):
        # Training Hyperparams
        params.epochs     = 15
        params.batch_size = None

        # Data Set
        params.dataset_mode = "keras"
        params.dataset      = keras.datasets.fashion_mnist
        params.class_names  = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        params.dataset      = None
        params.labels_path  = None

    else:
        default_configs(params, config)

def default_configs(params, config):
    if(config == "CatDog"):
        params.batch_size = 128
        params.epochs = 5
        params.dataset_mode = "files"
        params.dataset = 'C:\\Users\\manes\\.keras\\datasets\\cats_and_dogs_filtered'
        params.class_names = ['cats', 'dogs']
        params.source_shape = (150, 150)  # HW
        params.source_channels = 3

    elif(config == "FashionMnist"):
        params.batch_size = None
        params.epochs = 2
        params.dataset_mode = "keras"
        params.dataset = keras.datasets.fashion_mnist
        params.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    else:
        raise NameError("Invalid Config Name")


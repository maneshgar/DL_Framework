from tensorflow import keras

class InputManager(object):

    # parameters to be set by the user
    def set_params(self):
        self.dataset = keras.datasets.fashion_mnist
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                       'Ankle boot']

    # parameters to be set automatically
    def init(self):
        self.source_shape = None
        self.num_classes = len(self.class_names)


    def __init__(self, name):
        self.set_params()
        self.init()


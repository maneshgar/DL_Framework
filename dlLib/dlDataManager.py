# Data Manager
from stdLib import stdDisp as disp
from stdLib import stdUtil as util

# To represent a dataset
class DataSet(object):
    def __init__(self):
        # self.name = name
        self.train_images = None
        self.train_labels = None

        self.dev_images = None
        self.dev_labels = None

        self.test_images = None
        self.test_labels = None

# To Load data
class DataLoader(object):

    # update global params
    def update_params(self, p):
        p.source_shape = self.dataset.train_images.shape[1:3]

    def __init__(self, dataset, params):

        # Init and Load data
        self.dataset = DataSet()
        (self.dataset.train_images, self.dataset.train_labels), (self.dataset.test_images, self.dataset.test_labels) = dataset.load_data()

        # Normalize data
        self.dataset.train_images = util.normalize(self.dataset.train_images)
        self.dataset.test_images = util.normalize(self.dataset.test_images)

        # Show samples
        titles = []
        for i in range(25):
            titles.append(params.class_names[self.dataset.train_labels[i]])
        disp.gridShow(self.dataset.train_images[0:25], 5, 5, 5, 5, titles)

        # update global params
        self.update_params(params)





# Data Manager
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from stdLib import stdDisp as disp
from stdLib import stdUtil as util

# To represent a dataset
class DataSet(object):
    def __init__(self):
        # self.name = name
        self.train_images = None
        self.train_labels = None

        self.validation_images = None
        self.validation_labels = None

        self.test_images = None
        self.test_labels = None

# To Load data
class DataLoader(object):

    # update global params
    def update_params(self, p):
        p.source_shape = self.dataset.train_images.shape[1:3]

    def __init__(self, params):

        self.dataset = DataSet()

        if(params.dataset_mode == "files"):
            train_dir = os.path.join(params.dataset, 'train')
            validation_dir = os.path.join(params.dataset, 'validation')
            test_dir = os.path.join(params.dataset, 'test')

            num_class_tr = []
            print("training images:")
            for i,class_name in enumerate(params.class_names):
                train_class_dir = os.path.join(train_dir, class_name)  # directory of a pictures of each class
                num_class_tr.append(len(os.listdir(train_class_dir)))
                print('.{0}:\t {1}'.format(class_name, num_class_tr[i]))

            num_class_val = []
            print("validation images")
            for i,class_name in enumerate(params.class_names):
                validation_class_dir = os.path.join(validation_dir, class_name)
                num_class_val.append(len(os.listdir(validation_class_dir)))
                print('.{0}:\t {1}'.format(class_name, num_class_val[i]))

            params.total_train_images = sum(num_class_tr)
            params.total_validation_images = sum(num_class_val)
            print("--")
            print("Total training images:", params.total_train_images)
            print("Total validation images:", params.total_validation_images)

            train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
            validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

            self.dataset.train_images = train_image_generator.flow_from_directory(batch_size=params.batch_size,
                                                                           directory=train_dir,
                                                                           shuffle=True,
                                                                           target_size=params.source_shape,
                                                                            classes=params.class_names,
                                                                           class_mode='categorical')

            self.dataset.validation_images = validation_image_generator.flow_from_directory(batch_size=params.batch_size,
                                                                                           directory=validation_dir,
                                                                                           target_size=params.source_shape,
                                                                                           classes=params.class_names,
                                                                                           class_mode='categorical')

        elif(params.dataset_mode == "keras"):
            # Init and Load data
            (self.dataset.train_images, self.dataset.train_labels), (self.dataset.validation_images, self.dataset.validation_labels) = params.dataset.load_data()

            # Normalize data
            self.dataset.train_images = util.normalize(self.dataset.train_images)
            self.dataset.validation_images = util.normalize(self.dataset.validation_images)

            # update global params
            self.update_params(params)

            # Show samples
            titles = []
            for i in range(25):
                titles.append(params.class_names[self.dataset.train_labels[i]])
            disp.gridShow(self.dataset.train_images[0:25], 5, 5, 5, 5, titles)
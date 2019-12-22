from tensorflow import keras
from dlLib import dlConfig
###################################################################################
#####################   InputManager    ###########################################
###################################################################################
class InputManager(object):

    ###################################################################################
    ###################################################################################
    # parameters to be set automatically
    def init(self):
        self.num_classes = len(self.class_names)

    ###################################################################################
    ###################################################################################
    def __init__(self, name):
        self.source_shape = None
        self.source_channels = None
        self.total_train_images = None
        self.total_validation_images = None

        self.ConfigName = name
        dlConfig.set_params(self, name)
        self.init()
        self.parse()



    def parse(self):
        _dataset_mode = ["keras", "files"]
        if self.dataset_mode not in _dataset_mode:
            raise InvalidArg(self.dataset_mode, _dataset_mode)




class InvalidArg(Exception):
    """Raised when input argument is invalid

    Attributes:
        input argument -- user entered value
        validArgs -- list of valid input values
        message -- optional explanation of why the specific value is not allowed
    """

    def __init__(self, value, validArgs=None, message=None):
        print("*************************************")
        print("Invalid argument/parameter value \'{0}\'".format(value))
        print("List of valid arguments:")
        print(validArgs)
        if message is not None:
            print("message")
        print("*************************************")


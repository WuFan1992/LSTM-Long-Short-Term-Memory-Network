import numpy as np

# define the class LSTMLayer and initialization

class LSTMLayer(object):

    def _init_(self, input_width,state_width,learning_rate):

        self.input_width = input_width

        self.state_width = state_width

        self.learning_rate = learning_rate

        self.f_list = self.init_vector()

        




    def init_vector():

        return np.zeros((self.state_width,0))

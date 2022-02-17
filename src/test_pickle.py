# Step 1
import pickle
import numpy as np

class test():
    def __init__(self):
        self.config_dictionary = {1,2,3}
        self.world = np.array([[1,2,3],[4,5,6]])

with open('config.dictionary', 'wb') as config_dictionary_file:
  pickle.dump(test(), config_dictionary_file)

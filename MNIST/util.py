import numpy as np
from mnist import MNIST
from os.path import expanduser
home = expanduser("~")+'\Desktop\FUndAI\python-mnist\data'
mndata= MNIST(home)
mndata.gz = True
train_image, train_label = mndata.load_training() 
test_image, test_label = mndata.load_testing() 

class Util:
    def __init__(self):
        self.train = np.asarray([np.reshape(x, (784)) for x in train_image]).astype('float64') 
        self.train_label = np.asarray(train_label)
        self.test = np.asarray([np.reshape(x, (784)) for x in test_image]).astype('float64') 
        self.test_label = np.asarray(test_label)
        self.n_class = list(range(10)) 
        print('Data successfully loaded')
        
    def accuracy(self, actual, predict):
        print('Tests are on the way')
        correct = np.sum(actual == predict)
        total = len(actual)
        acc = correct / total
        print('Execution finished')
        print('Has an Accuracy of about: {}%'.format(acc*100))
        return acc
        


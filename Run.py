import scipy.io
import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt

try:
    os.system("clear")
except:
    pass


data = scipy.io.loadmat('MNISTmini.mat')

print(data.keys())

training_set = data['train_fea1']
training_label = data['train_gnd1']
testing_set = data['test_fea1']
testing_label = data['test_gnd1']

print(training_set.shape)
print(testing_set.shape)

for entry in set(training_label):
    print(f"\nIndex of {entry} is {training_label.index(entry)} \n")

# Set num image
num = 6001

# Print out the label of the image
print(training_label[num])

# Visualize an image from the dataset

plt.imshow(training_set[num].reshape(10,10), cmap='gray')
plt.show()

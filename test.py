#mytest.py
import mnist_loader
import network
import matplotlib.pyplot as plt
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

net = network.Network([784, 30, 10])
net.SGD(training_data, 20, 10, 3.0, test_data=test_data)
# Libraries
import csv
import numpy
from PIL import Image, ImageShow

test_qs = []

with open("mnist_test_data/mnist_test.csv") as csv_file:
    test_data = list(csv.reader(csv_file, delimiter=','))[1:]
    for test in test_data:
        test_qs.append(test)


def sigmoid(z):
    return 1/(1 + numpy.exp(-z))

class NeuralNetwork(object):
    """
    Neural Network: Input layer | Intermediate layer(s) | Output layer

    Network size =: [ size of input layer, size(s) of intermediate layer(s) , size of output layer ]
    Input values are represented by:
        - a unidimensional array of an image's pixel values
        - a label of the image (index 0).

    For the MNIST dataset:
        - Input layer is 28 pixels * 28 pixels = 784 (total number of pixels);
        - Output layer is 10 (total number of labels).

    Number and size of the intermediate layers are to be determined.
    """
    def __init__(self, input, arch):
        self.input = input
        self.ilabel = int(input[0])
        self.ilayer = numpy.array(input[1:], dtype=numpy.uint8)/255
        # ----------------
        self.arch = arch
        self.mlayers = [numpy.zeros(size) for size in arch]
        # ----------------
        self.olayer = numpy.zeros(10)
        # ----------------
        self.network = [len(self.ilayer), *arch, len(self.olayer)]
        # ----------------
        self.biases = [numpy.random.randn(j) for j in self.network[1:]]
        self.weights = [numpy.random.randn(i, j) for i, j in zip(self.network[:-1], self.network[1:])]
        # ----------------
        self.clayers = [self.ilayer, *self.mlayers, self.olayer]

    def forward(self):
        for i in range(len(self.clayers) - 1):
            update = sigmoid(numpy.matmul(self.clayers[i], self.weights[i]) + self.biases[i])
            numpy.put(self.clayers[i + 1], numpy.indices((len(self.clayers[i + 1]),))[0], update)
        return self.clayers

nn = NeuralNetwork(test_qs[0], [16, 16])

print(nn.ilabel)

t = numpy.zeros(10)
t[nn.ilabel] = 1.0
print(t)

f = nn.forward()
print(f[-1])

print(sum((t-f[-1])**2))



if __name__ == '__main__':
    print('')
"""
input nodes
intermediate layers of nodes
output nodes


weights and biases
cost function to be minimized

training data
testing data

"""
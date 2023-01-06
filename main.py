# Libraries
import csv
import numpy
import matplotlib.pyplot as plt

tests = []

with open("mnist_test_data/mnist_test.csv") as csv_file:
    test_data = list(csv.reader(csv_file, delimiter=','))[1:]
    for test in test_data:
        tests.append(test)

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
        self.input = input                  # len = 785
        self.ilabel = int(input[0])
        self.ilayer = numpy.array(input[1:], dtype=numpy.uint8)/255             # (784,) shaped of input array (1x784)
        # ----------------
        # architecture of the neural network
        self.arch = arch
        self.mlayers = [numpy.zeros(size) for size in arch]
        # ----------------
        self.olayer = numpy.zeros(10)
        self.target = numpy.zeros(10)
        self.target[self.ilabel] = 1
        # ----------------
        self.network = [len(self.ilayer), *arch, len(self.olayer)] # 4
        # ----------------
        # fix random sample
        numpy.random.seed(0)
        # (3,) shaped array of arrays obeying the network architecture: [(16,), (16,), (10,)]
        # b[layer index][layer value index] | 0 to number of layers - 2 = [0, 1 ,2]
        self.biases = numpy.array([numpy.random.randn(j) for j in self.network[1:]], dtype=object)
        # (3,) shaped array of matrices obeying the network architecture: [(784, 16), (16, 16), (16, 10)]
        # w[layer index][previous layer index as row][layer index as column] | 0 to number of layers - 2 = [0, 1 ,2]
        self.weights = numpy.array([numpy.random.randn(i, j) for i, j in zip(self.network[:-1], self.network[1:])], dtype=object)
        # ----------------
        self.zlayers = [self.ilayer, *self.mlayers, self.olayer] # 4
        self.alayers = [self.ilayer, *self.mlayers, self.olayer] # 4
        # size of the network: input, intermediate layers, output | 2 + number of intermediate layers = 4
        self.size = len(self.network)

    def forward(self, weights, biases):
        for i in range(self.size - 1): # i = 0, 1, 2
            z = numpy.matmul(self.alayers[i], weights[i]) + biases[i]
            # (1x784) * (784x16) + (1x16) = (1x16) | (1x16) * (16x16) + (1x16) = (1x16) | (1x16) * (16x10) + (1x10) = (1x10)
            numpy.put(self.zlayers[i + 1], numpy.indices((len(self.zlayers[i + 1]),))[0], z)
            a = sigmoid(z)
            numpy.put(self.alayers[i + 1], numpy.indices((len(self.alayers[i + 1]),))[0], a)
        return self.alayers

    def cost(self):
        return sum((self.alayers[self.size - 1] - self.target) ** 2)

    def diffs(self):
        # (3,) shaped array of matrices obeying the network architecture: [(784, 16), (16, 16), (16, 10)]
        # dw[layer index][previous layer index as row][layer index as column] | 0 to number of layers - 2 = [0, 1 ,2]
        dw = numpy.array([numpy.zeros((i, j)) for i, j in zip(self.network[:-1], self.network[1:])], dtype=object)
        # (3,) shaped array of arrays obeying the network architecture: [(16,), (16,), (10,)]
        # db[layer index][layer value index] | 0 to number of layers - 2 = [0, 1 ,2]
        db = numpy.array([numpy.zeros(j) for j in self.network[1:]], dtype=object)

        # dw[2]
        for i in range(dw[2].shape[0]): # 16 | (16, 10)
            numpy.put(dw[2][i],
                      numpy.indices((dw[2].shape[1],)),
                      numpy.outer(self.alayers[2],
                                  (-2) * numpy.multiply(numpy.multiply(self.alayers[3] - self.target,
                                                                       sigmoid(self.zlayers[3])
                                                                       ),
                                                        (1 - sigmoid(self.zlayers[3]))
                                                        )
                                  )[i]
                      )
        # dw[1]
        for i in range(dw[1].shape[0]): # 16 | (16, 16)
            numpy.put(dw[1][i],
                      numpy.indices((dw[1].shape[1],)),
                      self.alayers[1][i] * numpy.multiply(
                          numpy.multiply(sigmoid(self.zlayers[2]), (1 - self.zlayers[2])),
                          numpy.ndarray.sum(
                              (-2) * numpy.multiply(numpy.multiply(numpy.multiply(self.alayers[3] - self.target,
                                                                                  sigmoid(self.zlayers[3])
                                                                                  ),
                                                                   (1 - sigmoid(self.zlayers[3]))
                                                                   ),
                                                    self.weights[2]
                                                    ), axis=1
                              )
                          )
                      )
        # dw[0]
        for i in range(dw[0].shape[0]):
            numpy.put(dw[0][i],
                      numpy.indices((dw[0].shape[1],)),
                      self.alayers[0][i] * numpy.multiply(
                          numpy.multiply(sigmoid(self.zlayers[1]), (1 - self.zlayers[1])),
                          numpy.ndarray.sum(
                          numpy.multiply(numpy.ndarray.sum(
                              (-2) * numpy.multiply(numpy.multiply(numpy.multiply(self.alayers[3] - self.target,
                                                                                  sigmoid(self.zlayers[3])
                                                                                  ),
                                                                   (1 - sigmoid(self.zlayers[3]))
                                                                   ),
                                                    self.weights[2]
                                                    ), axis=1
                              ), numpy.multiply(numpy.multiply(sigmoid(self.zlayers[2]),
                                                               (1 - sigmoid(self.zlayers[2]))
                                                               ),
                                                self.weights[1])), axis = 1
                                                                   )))

        # db[2]
        for i in range(1):
            numpy.put(db[2],
                      numpy.indices((db[2].shape[0],)),
                      (-2) * numpy.multiply(numpy.multiply(self.alayers[3] - self.target,
                                                           sigmoid(self.zlayers[3])
                                                           ),
                                            (1 - sigmoid(self.zlayers[3]))
                                            )
                      )
        # db[1]
        for i in range(1):
            numpy.put(db[1],
                      numpy.indices((db[1].shape[0],)),
                      numpy.multiply(
                          numpy.multiply(sigmoid(self.zlayers[2]), (1 - self.zlayers[2])),
                          numpy.ndarray.sum(
                              (-2) * numpy.multiply(numpy.multiply(numpy.multiply(self.alayers[3] - self.target,
                                                                                  sigmoid(self.zlayers[3])
                                                                                  ),
                                                                   (1 - sigmoid(self.zlayers[3]))
                                                                   ),
                                                    self.weights[2]
                                                    ), axis=1
                          )
                      )
                      )
        # db[0]
        for i in range(1):
            numpy.put(db[0],
                      numpy.indices((db[0].shape[0],)),
                      numpy.multiply(
                          numpy.multiply(sigmoid(self.zlayers[1]), (1 - self.zlayers[1])),
                          numpy.ndarray.sum(
                              numpy.multiply(numpy.ndarray.sum(
                                  (-2) * numpy.multiply(numpy.multiply(numpy.multiply(self.alayers[3] - self.target,
                                                                                      sigmoid(self.zlayers[3])
                                                                                      ),
                                                                       (1 - sigmoid(self.zlayers[3]))
                                                                       ),
                                                        self.weights[2]
                                                        ), axis=1
                              ), numpy.multiply(numpy.multiply(sigmoid(self.zlayers[2]),
                                                               (1 - sigmoid(self.zlayers[2]))
                                                               ),
                                                self.weights[1])), axis=1
                          )))

        return dw, db

if __name__ == '__main__':
    nn = NeuralNetwork(tests[0], [16, 16])

    pixels = nn.ilayer.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

    nn.forward(nn.weights, nn.biases)
    print(nn.cost())
    print(nn.olayer, nn.ilabel)
    plt.plot(numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), nn.olayer)
    plt.show()
    dw, db = nn.diffs()
    nn.forward(nn.weights + dw, nn.biases + db)
    print(nn.cost())
    print(nn.olayer, nn.ilabel)
    plt.plot(numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), nn.olayer)
    plt.show()
    dw1, db1 = nn.diffs()
    nn.forward(nn.weights + dw + dw1, nn.biases + db + db1)
    print(nn.cost())
    print(nn.olayer, nn.ilabel)
    plt.plot(numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), nn.olayer)
    plt.show()
    dw2, db2 = nn.diffs()
    nn.forward(nn.weights + dw + dw1 + dw2, nn.biases + db + db1 + db2)
    print(nn.cost())
    print(nn.olayer, nn.ilabel)
    plt.plot(numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), nn.olayer)
    plt.show()
    dw3, db3 = nn.diffs()
    nn.forward(nn.weights + dw + dw1 + dw2 + dw3, nn.biases + db + db1 + db2 + db3)
    print(nn.cost())
    print(nn.olayer, nn.ilabel)
    plt.plot(numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), nn.olayer)
    plt.show()

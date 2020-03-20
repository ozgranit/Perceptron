import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from tabulate import tabulate
import sgd

# returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
def perceptron(data, labels):
    # Start with the all-zeroes weight vector
    w = np.zeros(data.shape[1])
    data = normalize(data)
    # go over data
    for i in range(data.shape[0]):
        x = data[i]
        y = labels[i]
        # calc w dot x
        # if mistake
        if np.dot(w, x) > 0:
            prediction = 1
        else:
            prediction = -1
        if prediction != y:
            w = w + x * y
    return w


def calc_accuracy(w):
    err = 0
    for i in range(len(test_data)):
        if np.dot(w, test_data[i]) > 0:
            prediction = 1
        else:
            prediction = -1
        if prediction != test_labels[i]:
            err += 1
    accuracy = 1 - err / len(test_data)
    return accuracy


# returns two misclassified images, one false positive one false negative
def get_wrong_images(w):
    image1, image2 = 0, 0
    got_one = False
    got_two = False
    while True:
        i = np.random.randint(0, len(test_data) - 1)
        if np.dot(w, test_data[i]) > 0:
            if test_labels[i] < 0:
                image1 = test_data[i]
                got_one = True
        else:
            if test_labels[i] > 0:
                image2 = test_data[i]
                got_two = True
        if got_one and got_two:
            break
    return image1, image2


if __name__ == '__main__':
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = sgd.helper()
    w = perceptron(train_data, train_labels)
    """
    # show perceptron weight vector as image 
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.title("w_star")
    plt.show()"""
    print("The accuracy of the classifier trained on the full training set is " + str(calc_accuracy(w)))
    # show example of mistakes of perceptron
    im1, im2 = get_wrong_images(w)
    plt.imshow(np.reshape(im1, (28, 28)), interpolation='nearest')
    plt.title("false 8")
    plt.show()
    plt.imshow(np.reshape(im2, (28, 28)), interpolation='nearest')
    plt.title("false 0")
    plt.show()
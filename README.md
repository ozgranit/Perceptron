# Perceptron
handwritten digits classifier (using the MNIST dataset, separates '0' and '8')

The MNIST dataset consists of images of handwritten digits, 
along with their labels. Each image has 28×28 pixels, where each pixel is in grayscale
scale, and can get an integer value from 0 to 255. Each label is a digit between 0 and 9. The
dataset has 70,000 images. Althought each image is square, we treat it as a vector of size 28×28 = 784.

**In the file sgd.py there is a helper function**. this repository uses it. The function reads the examples
labelled 0, 8 and returns them with 0−1 labels. 

### Perceptron Algorithm:
Accepts as input:
- a set of images
- a vector of labels corresponding to the images 

returns a hyperplane on the MNIST dataset.

we show example of images misclassified by perceptron, 
and print the perceptron weight vector as an image,
showing what pixels the algorithm uses to classify.

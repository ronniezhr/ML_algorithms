# ML_algorithms

In folder knn, I use k-nearest-neighbor algorithm to implement digit image recognition. The data set is a reduced version
of the MNIST handwritten digit data set. Each image of a digit is a 28×28 pixel image. Features have been pre-extracted,
using a very simple scheme: each pixel is its own feature, so 28^2 = 784 features in total. The value of a feature is
the intensity of that corresponding pixel, normalized to be in the range 0..1. The images are preprocessed and vectorized
into feature vectors. The image folder provides a visualization of original images.

In folder RandomForest, I use random forest algorithm to build a classifier that, given an email, correctly predicts
whether or not this email is spam or not. 57 useful features are selected and preprocessed extracted from the emails. 0
stands for non-spam and 1 stands for spam.

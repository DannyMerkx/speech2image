# speech2image
This project is an implementation of a speech to image network which is trained to map images and spoken captions of those images to the same vector space.

Important notice:
The code is my own work, using python, theano, keras and lasagne. However some of the ideas and data are not:

get the vgg16 pretrained weights here https://github.com/fchollet/deep-learning-models/releases , this is NOT my work, my thanks to F. Chollet who made these availlable. I used the following weights from that page : vgg16_weights_th_dim_ordering_th_kernels.h5

The speech2image neural networks were originally introduced by D. Harwath and J. Glass  (2016) in the paper called: Unsupervised Learning of Spoken Language with Visual Context. The basic neural network structure (the one in speech2im_net.py) and the use of the l2norm hinge loss function is a theano based reproduction of the ideas and work described in that paper.


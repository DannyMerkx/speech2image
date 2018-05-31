# speech2image
This project is an implementation of a speech to image network which is trained to map images and captions of those images to the same vector space. This project contains networks for tokenized captions, raw text (character based prediction) and spoken captions. 

Important notice:
The code is my own work, using python, theano, lasagne and Pytorch. However some of the ideas and data are not:

The pretrained networks included in PyTorch (e.g. vgg16 vgg19 and resnet) are not trained or made by me but freely availlable in PyTorch.
Please cite the original creators of any pretrained network you use. 

The speech2image neural networks were originally introduced by D. Harwath and J. Glass  (2016) in the paper called: Unsupervised Learning of Spoken Language with Visual Context. The basic neural network structure (the one in speech2im_net.py) and the use of the l2norm hinge loss function is a theano and PyTorch based reproduction of the ideas and work described in that paper.

N.B. I switched from Theano to PyTorch early on in the project. The Theano part was abandoned in the early stages and while (it was at some point) working not as refined as the PyTorch implementation. 

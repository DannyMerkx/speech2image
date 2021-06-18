# speech2image
This project is an implementation of a speech to image network which is trained to map images and captions of those images to the same vector space. This project contains networks for tokenized captions, raw text (character based prediction) and spoken captions. 

Important notice:
The code is my own work, using python and Pytorch. However some of the ideas and data are not:

The pretrained networks included in PyTorch (e.g. vgg16 vgg19 and resnet) are not trained or made by me but are freely available in PyTorch.
Please cite the original creators of any pretrained network you use. 

The speech2image neural networks were originally introduced by D. Harwath and J. Glass  (2016) in the paper called: Unsupervised Learning of Spoken Language with Visual Context. The basic neural network structure (the one in speech2im_net.py) and the use of the l2norm hinge loss function is PyTorch based reproduction of the ideas and work described in that paper.

The Interspeech19 branch contains the code as it was used in our Interspeech2019 paper.
I continued work on this model and the most recent version of the code has large differences from the Interspeech 2019 version. Please make sure to 
use the correct branch if you want to reproduce the results in the paper as old models and analysis code might not work on the newest 
version of my code.

Feel free to use this repo in your own work if you cite my paper and please also consider citing the relevant papers used in this repo. 

Citation: 

@inproceedings{Merkx2019, author={Danny Merkx and Stefan L. Frank and Mirjam Ernestus}, title={{Language Learning Using Speech to Image Retrieval}}, year=2019, booktitle={Proc. Interspeech 2019}, pages={1841--1845}, doi={10.21437/Interspeech.2019-3067}, url={http://dx.doi.org/10.21437/Interspeech.2019-3067} }


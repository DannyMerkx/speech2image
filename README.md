# speech2image 
This branch contains the code at the state it was in when submitting our Interspeech 2021 paper and can be used to replicate the results presented therein.

This project is an implementation of a speech to image network which is trained to map images and captions of those images to the same vector space. This project contains networks for tokenized captions, raw text (character based prediction) and spoken captions. 

Important notice:
The code is my own work, using python and Pytorch. However some of the ideas and data are not:

The pretrained networks included in PyTorch (e.g. vgg16 vgg19 and resnet) are not trained or made by me but are freely available in PyTorch.
Please cite the original creators of any pretrained network you use. 

The speech2image neural networks were originally introduced by D. Harwath and J. Glass  (2016) in the paper called: Unsupervised Learning of Spoken Language with Visual Context. The basic neural network structure (the one in speech2im_net.py) and the use of the l2norm hinge loss function is PyTorch based reproduction of the ideas and work described in that paper.

Been doing a lot of work on implementing the Vector Quantization layers as used Harwath, D., Hsu, W.-N., & Glass, J. (2019). Learning Hierarchical Discrete Linguistic Units from Visually-Grounded Speech, 000, 1–19. Retrieved from http://arxiv.org/abs/1911.09602. There is now a working reimplementation of the convolutional architecture in this paper + a working addition of VQ layers to my own RNN based model. 

The VQ layer is so far based on the code from Zalando research, still working on getting my own implementation working.

Feel free to use this repo in your own work, please consider citing my paper and the relevant papers used in this repo. 
Citation: 
@inproceedings{Merkx2019,
  author={Danny Merkx and Stefan L. Frank and Mirjam Ernestus},
  title={{Language Learning Using Speech to Image Retrieval}},
  year=2019,
  booktitle={Proc. Interspeech 2019},
  pages={1841--1845},
  doi={10.21437/Interspeech.2019-3067},
  url={http://dx.doi.org/10.21437/Interspeech.2019-3067}
}

# speech2image
This project is an implementation of a (spoken) caption to image network which is trained to map images and captions of those images to the same vector space. This project contains networks for tokenized captions, raw text (character based prediction) and spoken captions. 

Important notice:
The code is my own work, using python and Pytorch. However some of the ideas and data are not:

The pretrained networks included in PyTorch (e.g. vgg16 vgg19 and resnet) are not trained or made by me but are freely available in PyTorch.
Please cite the original creators of any pretrained network you use. 

The speech2image neural networks were originally introduced by D. Harwath and J. Glass  (2016) in the paper called: Unsupervised Learning of Spoken Language with Visual Context. The basic neural network structure (the one in speech2im_net.py) and the use of the l2norm hinge loss function is PyTorch based reproduction of the ideas and work described in that paper.

This branch is associated with our latest paper where we compare grounded word embeddings to text-based word embeddings:

Distributional semantic models capture word-level meaning that is useful in many natural language processing tasks and have even been shown to capture cognitive aspects of word meaning. The majority of these models are purely text based, even though the human sensory experience is much richer. In this paper we create visually grounded word embeddings by combining English text and images and compare them to popular text-based methods, to see if visual information allows our model to better capture cognitive aspects of word meaning. Our analysis shows that visually grounded embedding similarities are more predictive of the human reaction times in a large priming experiment than the purely text-based embeddings. The visually grounded embeddings also correlate well with human word similarity ratings. Importantly, in both experiments we show that the grounded embeddings account for a unique portion of explained variance, even when we include text-based embeddings trained on huge corpora. This shows that visual grounding allows our model to capture information that cannot be extracted using text as the only source of information. 

This branch contains the state of the code at time of submission.

Feel free to use this repo in your own work, please consider citing my papers and the relevant papers used in this repo. 
Citation: 

@article{Merkx2022, author={Danny Merkx and Stefan L. Frank and Mirjam Ernestus}, title={{Seeing the advantage: visually grounding word embeddings to better capture human semantic knowledge}}, year=2022, journal={Arxiv preprint}, pages={1-10}, url={https://arxiv.org/abs/2202.10292} }

@inproceedings{Merkx2019,
  author={Danny Merkx and Stefan L. Frank and Mirjam Ernestus},
  title={{Language Learning Using Speech to Image Retrieval}},
  year=2019,
  booktitle={Proc. Interspeech 2019},
  pages={1841--1845},
  doi={10.21437/Interspeech.2019-3067},
  url={http://dx.doi.org/10.21437/Interspeech.2019-3067}
}

@inproceedings{Merkx2021, author={Danny Merkx and Stefan L. Frank and Mirjam Ernestus}, title={{Semantic Sentence Similarity: Size Does Not Always Matter}}, year=2021, booktitle={Proc. Interspeech 2021}, pages={1-5}, url={https://arxiv.org/abs/2106.08648} }



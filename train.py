#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:25:48 2019

@author: Lily Nway Nway Aung
"""

import torch
torch.cuda.current_device()
torch.cuda.device(0)
torch.cuda.device_count()
print('count',torch.cuda.device_count())
torch.cuda.is_available()
torch.cuda.get_device_name(0)

import sys
sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO
#!pip install nltk
import nltk
nltk.download('punkt')
from data_loader import get_loader
from torchvision import transforms
import nltk

# Define a transform to pre-process the training images.
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])
        # numpy image: H x W x C
        # torch image: C X H X W
# Set the minimum word count threshold.
vocab_threshold = 5

# Specify the batch size.
batch_size = 10

# Obtain the data loader.
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False)
sample_caption = 'A person doing a trick on a rail while riding a skateboard.'
sample_caption = []
sample_tokens = nltk.tokenize.word_tokenize(str(sample_caption).lower())
print(sample_tokens)
start_word = data_loader.dataset.vocab.start_word
print('Special start word:', start_word)
sample_caption.append(data_loader.dataset.vocab(start_word))
print(sample_caption)
sample_caption.extend([data_loader.dataset.vocab(token) for token in sample_tokens])
print(sample_caption)
end_word = data_loader.dataset.vocab.end_word
print('Special end word:', end_word)

sample_caption.append(data_loader.dataset.vocab(end_word))
print(sample_caption)


import torch

sample_caption = torch.Tensor(sample_caption).long()
print(sample_caption)
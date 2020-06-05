import argparse

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, models

import numpy as np

from PIL import Image

import json
import os
import random

from cat_names import cat_names
from process_image import process_image
from load_checkpoint import load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='3')
    parser.add_argument('--filepath', dest='filepath', default='flowers/test/1/image_06743.jpg')
    parser.add_argument('--names', dest='names', default='cat_to_name.json')
    parser.add_argument('--gpu', action="store", default="gpu")
    return parser.parse_args()


def predict(image_path, model, gpu, topk=3):
    
    if(gpu == 'gpu'):
        model = model.cuda()
    else:
        model.cpu()
        
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    if(gpu == 'gpu'):
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with toch.no_grad():
            output = model.forward(img_torch)
        
    probability = F.softmax(output.data,dim=1) 
    
    probs = np.array(probability.topk(topk)[0][0])
    
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [np.int(index_to_class[each]) for each in np.array(probability.topk(topk)[1][0])]
    
    return probs, top_classes


def main():
    args = parse_args()
    model = load_checkpoint(args.checkpoint)
    gpu = args.gpu
    cat_to_name = cat_names(args.names)
    img_path = args.filepath
    probs, classes = predict(img_path, model, gpu, int(args.top_k))
    labels = [cat_to_name[str(index)] for index in classes]
#     probability = probs
    print(labels, classes)
    
    
if __name__== '__main__':
    main()
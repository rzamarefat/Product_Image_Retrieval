from scipy import spatial
from EmbGenerator import EmbGenerator
import cv2
import pickle5 as pickle
from numpy import genfromtxt
import os
import cv2
from glob import glob
import pandas as pd

def calc_cosine_distance(emb_one, emb_two):
    distance = spatial.distance.cosine(emb_one, emb_two)
    return distance

def preprocess(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    return image

def prepare_medoids(medoids_path):
    with open(medoids_path, 'rb') as handle:
        products_medoids = pickle.load(handle)
        
    return products_medoids

def create_target_embs_dataset(path_to_embs_ds, target_product):
    ds = {}
    print(path_to_embs_ds)
    for product in sorted(os.listdir(path_to_embs_ds)):
        if product == target_product:
            for emb_name in sorted(os.listdir(os.path.join(path_to_embs_ds, product))):
                abs_path_to_emb_file = os.path.join(path_to_embs_ds, product, emb_name)
                emb_data = genfromtxt(abs_path_to_emb_file, delimiter=',')
                ds[emb_name.split(".")[0]] = emb_data
    return ds

def sort_ranking(lst): 
    return sorted(lst, key = lambda x: x[1])
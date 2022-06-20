import pickle5 as pickle
import cv2
import os
import numpy as np
from sklearn_extra.cluster import KMedoids
from numpy import genfromtxt
import argparse
from glob import glob
import pandas as pd





parser = argparse.ArgumentParser(description='Embedding Extracion ')

parser.add_argument('--parent-path-to-embs', type=str, required=True, help='Parent path to where the embs will be stored')
parser.add_argument('--abs-path-to-save-medoids', type=str, required=True, help='Path to save medoids')
parser.add_argument('--abs-path-to-gt', type=str, required=True, help='Path to Fashion dataset emb')

args = parser.parse_args()

embs_path = [file for file in sorted(glob(os.path.join(args.parent_path_to_embs, "*.csv")))]
df = pd.read_csv(args.abs_path_to_gt)
categories = list(df['SubCategory'])
img_names = list(df['Image'])


products_embeddings_db = {}

for category in categories:
    products_embeddings_db[category] = []

for emb_path in sorted(embs_path):
    emb_name = emb_path.split("/")[-1].split(".")[0]
    category = categories[img_names.index(emb_name+".jpg")]
    emb_data = genfromtxt(emb_path, delimiter=',')
    products_embeddings_db[category].append(emb_data)


num_clusters = 3

medoids_db = {}
for name in products_embeddings_db.keys():
    each_product_embs = products_embeddings_db[name]
    kmed = KMedoids(n_clusters=num_clusters, metric='euclidean', method='alternate', init='heuristic', max_iter=300, random_state=4)
    kmed.fit(each_product_embs)
    medoids_db[name] = kmed.cluster_centers_

with open(args.abs_path_to_save_medoids, 'wb') as handle:
    pickle.dump(medoids_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
from scipy import spatial
from EmbGenerator import EmbGenerator
import cv2
import pickle5 as pickle
from numpy import genfromtxt
import os
import cv2
from glob import glob
import pandas as pd
import shutil as sh

class ProductImageRetriever():
    def __init__(self, path_to_medoids="./fashion_medoids.pickle", path_to_csv_meta="./data/fashion.csv", path_to_embs_root="./fashion_embs"):
        
        self.path_to_medoids = path_to_medoids
        self.path_to_csv_meta = path_to_csv_meta
        self.path_to_embs_root = path_to_embs_root



        self.gt_df = pd.read_csv(self.path_to_csv_meta)
        self.images_names = list(self.gt_df['Image'])
        self.images_path = [file for file in sorted(glob("./data/*/*/*/*/*"))]
        self.categories = list(self.gt_df['SubCategory'])

        
        self.emb_gen = EmbGenerator()
        self.embs_ds = self.__get_embs()
        self.products_medoids = self.__prepare_medoids(path_to_medoids)

        
        

    def __get_embs(self):
        embs_pathes = [file for file in sorted(glob(os.path.join((self.path_to_embs_root), "*.csv")))]

        embs_ds = {}
        for emb in embs_pathes:
            emb_name = emb.split("/")[-1].split(".")[0] + ".jpg"
            cat = self.categories[self.images_names.index(emb_name)]
            emb_data = genfromtxt(emb, delimiter=',')

            embs_ds[emb_name] = (cat, emb_data)

        return embs_ds



        

    def __calc_cosine_distance(self, emb_one, emb_two):
        distance = spatial.distance.cosine(emb_one, emb_two)
        return distance

    def __preprocess(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        return image

    def __prepare_medoids(self, medoids_path):
        with open(medoids_path, 'rb') as handle:
            products_medoids = pickle.load(handle)
            
        return products_medoids

    def __find_target_category(self, query_emb):
        lowest_one = (float('inf'), None)
        for pm_name, pms in self.products_medoids.items():
            d = sum([self.__calc_cosine_distance(pm, query_emb) for pm in pms])

            if d < lowest_one[0]:
                lowest_one = (d, pm_name)


        return lowest_one



    def retrieve(self, img_path, number_to_fetch=3):
        query_img = self.__preprocess(img_path)
        query_emb = self.emb_gen.get_emb(img_path)

        target_cat = self.__find_target_category(query_emb)
        
        fetched_embs = {}

        for k,v in self.embs_ds.items():
            if v[0] == target_cat[1]:
                fetched_embs[k] = self.__calc_cosine_distance(query_emb, v[1])
            
        fetched_embs = dict(sorted(fetched_embs.items(), key=lambda item: item[1]))
        print(fetched_embs)
        for index, (img_name, _) in enumerate(fetched_embs.items()):
            if index == number_to_fetch:
                break

            src = list(filter(lambda x: x.split("/")[-1] == img_name, self.images_path))[0]
            dest = f"./fetched_items/{index}.jpg"
            sh.copy(src, dest)
        


if __name__ == "__main__":
    path_to_img = "/mnt/829A20D99A20CB8B/projects/github_projects/PIR/data/Apparel/Girls/Images/images_with_product_ids/2712.jpg"
    path_to_medoids = "/mnt/829A20D99A20CB8B/projects/github_projects/PIR/fashions_medoids.pickle"
    
    
    pir = ProductImageRetriever(path_to_medoids)
    
    pir.retrieve(path_to_img, number_to_fetch=5)

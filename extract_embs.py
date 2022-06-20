from EmbGenerator import EmbGenerator
from numpy import savetxt
import os
from tqdm import tqdm
import argparse
from glob import glob


parser = argparse.ArgumentParser(description='Embedding Extracion ')

parser.add_argument('--parent-path-to-images', type=str, required=True, help='Parent path to the gallery images')
parser.add_argument('--parent-path-to-embs', type=str, required=True, help='Parent path to where the embs will be stored')

args = parser.parse_args()

emb_gen = EmbGenerator()

images = [file for file in sorted(glob(os.path.join(args.parent_path_to_images, "*", "*","*","*","*.jpg")))]
print(f"=============> Number of images to extract embs for: {len(images)}")


if not(os.path.isdir(args.parent_path_to_embs)):
    os.mkdir(args.parent_path_to_embs)

if __name__ == "__main__":
    for img in tqdm(images):
        img_name = img.split("/")[-1].split(".")[0]
        abs_path_to_save = os.path.join(args.parent_path_to_embs, f"{img_name.split('.')[0]}.csv")
        emb = emb_gen.get_emb(img)
        savetxt(abs_path_to_save, emb, delimiter=',')
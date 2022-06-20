Download the Fashion Dataset and unzip it using the following command
```
unzip fashion-images.zip 
```
A folder named data containing two folders, named Apparel and Footwear and a csv file named fashion.csv, is created. This is the root directory of the dataset and the code is customized for the structure of it. So you do not need any modification.
 

Extract Embs:
```
python extract_embs.py --parent-path-to-images "/mnt/829A20D99A20CB8B/projects/github_projects/Product_Retrieval/data" --parent-path-to-embs /mnt/829A20D99A20CB8B/projects/img2vec/fashion_embs
```

Extract K-Medoids 
```
python extract_K_Medoids.py --parent-path-to-embs /mnt/829A20D99A20CB8B/projects/img2vec/fashion_embs --abs-path-to-save-medoids /mnt/829A20D99A20CB8B/projects/img2vec/fashions_medoids.pickle --abs-path-to-gt /mnt/829A20D99A20CB8B/projects/github_projects/Product_Retrieval/data/fashion.csv
```
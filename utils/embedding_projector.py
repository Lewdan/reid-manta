import pandas as pd
import cv2
import wandb
from pathlib import Path
wandb.init(project="catno_aug")


## defining a path - change to the location of desired embeddings + label csv
emb = 'catno_aug-db_emb.csv'
filepath_emb = Path().cwd() / 'reid-manta' /'examples'/'catno_aug' / 'db_embs' / emb

lbl = 'catno_aug-db_lbl.csv'
filepath_lbl = Path().cwd() /'reid-manta' /'examples'/'catno_aug' / 'db_embs' / lbl

## read in the filepath
df = pd.read_csv(filepath_emb)
ds = pd.read_csv(filepath_lbl)

df["target"] = pd.read_csv(filepath_lbl)["class"]
df["target"] = df.target.astype(str)    # your prediction of class (cat)
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]

## Reshape image to  16x16 as it needs to be divisable by total columns (in this case 257) - use below 3 lines to log embedding pixels
#df["image"] = df.apply(lambda row: wandb.Image(row.iloc[1:].values.reshape(16, 16) / 16.0), axis=1) ## This will log embedding 'fuzz' / 'static' not the actual cat image
#cols = df.columns.tolist()
#df = df[cols[-1:] + cols[:-1]]

## log image is grayscale (0), colour (1) and RGB (COLOUR_BGR2RGB)

#cv2.imread('/home/pichael/lewis/reid-cat/reid-manta/examples/catinitial/cropped/cat1.png', 0)

#df.apply(lambda: wandb.Image(cv2.imread('/home/pichael/lewis/reid-cat/reid-manta/examples/catinitial/cropped/cat1.png', 0)))

##df["image"] = df.apply(lambda row: wandb.Image(row[1:].values.reshape(16, 16) / 16.0), axis=1)

#im = cv2.imread(('lbl')["file"], 1)
#im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

#wandb.Image(cv2.imread('/home/pichael/lewis/reid-cat/reid-manta/examples/catinitial/cropped/cat1.png', 0)) ## 0 = grayscale, 1 = colour

#im = cv2.imread('/home/pichael/lewis/reid-cat/reid-manta/examples/catinitial/cropped/cat1.png', 1)
#im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


#To log to wandb
wandb.log({"catno_aug": df})
#wandb.log({'test_image_rgb': wandb.Image(im_rgb)})
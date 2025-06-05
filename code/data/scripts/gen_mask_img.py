import numpy as np
import os
import glob
import tqdm
from PIL import Image

mask_rate = 0.4

set_list = ['train', 'val', 'test']
# set_list = ['val']

for set_name in set_list:
    sample_dir = f'/mimer/NOBACKUP/groups/naiss2024-22-578/CREMA-D/AMST/80/visual/{set_name}_unsplit_imgs/Image-01-FPS'
    out_dir = f'/mimer/NOBACKUP/groups/naiss2024-22-578/CREMA-D/AMST/80/visual/{set_name}_unsplit_imgs_masked_{mask_rate}/Image-01-FPS'
    print("Processing set: {}".format(set_name))
    print("Input dir: {}".format(sample_dir))
    print("Output dir: {}".format(out_dir))
    
    os.makedirs(out_dir, exist_ok=True)
    
    # all folders in the sample_dir
    folders = os.listdir(sample_dir)
    
    for folder in tqdm.tqdm(folders, desc="Processing folders for {}".format(set_name)):
        folder_path = os.path.join(sample_dir, folder)
        out_folder_path = os.path.join(out_dir, folder)
        os.makedirs(out_folder_path, exist_ok=True)
        imgs = os.listdir(folder_path)
        for img in imgs:
            img_path = os.path.join(folder_path, img)
            img_out_path = os.path.join(out_folder_path, img)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img, dtype=np.uint8)
            mask = np.random.choice([0, 1], size=img.shape[:2], p=[1-mask_rate, mask_rate])
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
            img = np.where(mask == 1, mask, img).astype(np.uint8)
            img = Image.fromarray(img)
            img.save(img_out_path)
    
    print("Done processing set: {}".format(set_name))


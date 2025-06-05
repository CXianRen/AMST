import numpy as np
import os
import glob
import tqdm

mask_rate = 0.8

set_list = ['train', 'val', 'test']

for set_name in set_list:
    fbank_dir = '/mimer/NOBACKUP/groups/naiss2024-22-578/CREMA-D/AMST/80/audio/{}_fbank'.format(set_name)
    folder = fbank_dir.split('/')[-1]
    out_dir = os.path.join(fbank_dir, "../{}_masked_{}".format(folder, mask_rate))

    os.makedirs(out_dir, exist_ok=True)

    print("Processing set: {}".format(set_name))
    print("Mask rate: {}".format(mask_rate))
    print("Input directory: {}".format(fbank_dir))
    print("Output directory: {}".format(out_dir))

    for fbank_file in tqdm.tqdm(glob.glob(os.path.join(fbank_dir, "*.npy")), desc="Processing fbank files for {}".format(set_name)):
        fbank = np.load(fbank_file)
        mask = np.random.choice([0, 1], size=fbank.shape, p=[mask_rate, 1-mask_rate])
        masked_fbank = fbank * mask
        out_file = os.path.join(out_dir, os.path.basename(fbank_file))
        np.save(out_file, masked_fbank)

    print("Done processing set: {}".format(set_name))

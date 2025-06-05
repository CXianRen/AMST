# This file is used to configure the data path for different datasets.
# The data path is configured according to the dataset name.
# Why it looks a bit messy is because the historical reasons.
# We wrapped the data path in a class, and then we can use it in the code.

from types import SimpleNamespace
import os, json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"

IEMOCAP_CONFIG = {
    # raw data
    "data_root": '/mimer/NOBACKUP/groups/multimodal_learning/IEMOCAP_PROCESSED',
    "visual_feature_path": "/mimer/NOBACKUP/groups/multimodal_learning/IEMOCAP_PROCESSED/IMAGE_KEPT_2_PER_SEC",
    "audio_feature_path": "/mimer/NOBACKUP/groups/multimodal_learning/IEMOCAP_PROCESSED/fbank/",
    "text_feature_path": "/mimer/NOBACKUP/groups/multimodal_learning/IEMOCAP_PROCESSED/text_token/",
    # data files
    "stat_path": PROJECT_ROOT + "data/IEMOCAP/stat_iemocap.txt",
    "train_txt": PROJECT_ROOT + "data/IEMOCAP/iemocap_train.txt",
    "test_txt":  PROJECT_ROOT+ "data/IEMOCAP/iemocap_test.txt",
    "val_txt":   PROJECT_ROOT+ "data/IEMOCAP/iemocap_valid.txt"
}

DATA_PATH_CONFIG = {
    "AVE": {
        # raw data
        "data_root": '/mimer/NOBACKUP/groups/multimodal_learning/AVE_Dataset/AMST/',
        "visual_feature_path": None, # fill by get func
        "audio_feature_path": None,  # fill by get func
        # data files
        "stat_path": PROJECT_ROOT + "data/AVE/stat_ave.txt",
        "train_txt": PROJECT_ROOT + "data/AVE/my_train_ave.txt",
        "val_txt": PROJECT_ROOT + "data/AVE/my_val_ave.txt",
        "test_txt": PROJECT_ROOT + "data/AVE/my_test_ave.txt"
    },
    "CREMAD": {
        # raw data
        "data_root": '/mimer/NOBACKUP/groups/multimodal_learning/CREMA-D/AMST/80',
        "visual_feature_path": None,  # fill by get func
        "audio_feature_path": None,   # fill by get func
        # data files
        "stat_path": PROJECT_ROOT + "data/CREMAD/stat_cre.txt",
        "train_txt": PROJECT_ROOT + "data/CREMAD/80_train_cre.txt",
        "test_txt":  PROJECT_ROOT+ "data/CREMAD/10_test_cre.txt",
        "val_txt":   PROJECT_ROOT+ "data/CREMAD/10_val_cre.txt"
    },
    # IEMOCAP3 is the same as IEMOCAP, but with text features, for compatibility
    "IEMOCAP": IEMOCAP_CONFIG,
    "IEMOCAP3": IEMOCAP_CONFIG,
    "MVSA": {
        "data_root": '/mimer/NOBACKUP/groups/multimodal_learning/MVSA_Single/',
        "visual_feature_path": None,
        "text_feature_path": None,
        "stat_path": PROJECT_ROOT + "data/MVSA/stat_mvsa.txt",
        "train_txt": PROJECT_ROOT + "data/MVSA/my_train_mvsa.txt",
        "val_txt": PROJECT_ROOT + "data/MVSA/my_val_mvsa.txt",
        "test_txt": PROJECT_ROOT + "data/MVSA/my_test_mvsa.txt"
    },
    "URFUNNY": {
        "data_root": '/mimer/NOBACKUP/groups/multimodal_learning/UR-FUNNY/',
        "visual_feature_path": '/mimer/NOBACKUP/groups/multimodal_learning/UR-FUNNY/IMAGE_KEPT_1_PER_SEC/',
        "audio_feature_path": '/mimer/NOBACKUP/groups/multimodal_learning/UR-FUNNY/fbank/',
        "text_feature_path": '/mimer/NOBACKUP/groups/multimodal_learning/UR-FUNNY/text_token/roberta-base/',
        "stat_path": PROJECT_ROOT + "data/UR-FUNNY/ur_funny_stat.txt",
        "train_txt": PROJECT_ROOT + "data/UR-FUNNY/ur_funny_train.txt",
        "val_txt": PROJECT_ROOT + "data/UR-FUNNY/ur_funny_valid.txt",
        "test_txt": PROJECT_ROOT + "data/UR-FUNNY/ur_funny_test.txt"
    },
    "SSW60": {
        "data_root": '/mimer/NOBACKUP/groups/multimodal_learning/ssw60',
        "visual_feature_path": '/mimer/NOBACKUP/groups/multimodal_learning/ssw60/all_imgs/',
        "audio_feature_path": '/mimer/NOBACKUP/groups/multimodal_learning/ssw60/fbank/',
        "stat_path": None,
        "train_txt": PROJECT_ROOT + "data/SSW60/train.txt",
        "val_txt": PROJECT_ROOT + "data/SSW60/val.txt",
        "test_txt": PROJECT_ROOT + "data/SSW60/test.txt"
    }
}

def get_data_path_config(args, mode):     
    if args.dataset not in DATA_PATH_CONFIG:
        raise ValueError("Invalid dataset: {},"\
                          " please choose from {}".format(
                              args.dataset, DATA_PATH_CONFIG.keys()))

    dataset_cfg = DATA_PATH_CONFIG[args.dataset]
    dataset_cfg = SimpleNamespace(**dataset_cfg)
    if hasattr(args, "data_path") and args.data_path != "":
        dataset_cfg.data_root = args.data_path
        print("Using data path from args: ", dataset_cfg.data_root)
        
    if args.dataset == "AVE" or args.dataset == "CREMAD":
        dataset_cfg.visual_feature_path = os.path.join(
                        dataset_cfg.data_root, "visual/", 
                        '{}_unsplit_imgs/Image-01-FPS/'.format(mode))
        dataset_cfg.audio_feature_path = os.path.join(
                        dataset_cfg.data_root, "audio/", 
                        '{}_fbank/'.format(mode))
    elif args.dataset == "IEMOCAP" or args.dataset == "IEMOCAP3":
        pass
    elif args.dataset == "URFUNNY":
        pass
    elif args.dataset == "MVSA":
        dataset_cfg.visual_feature_path = os.path.join(
                        dataset_cfg.data_root, "visual/", 
                        '{}_imgs/'.format(mode))
        dataset_cfg.text_feature_path = os.path.join(
                        dataset_cfg.data_root, "text_token/",
                        "roberta-base/" 
                        '{}_token/'.format(mode))
    elif args.dataset == "SSW60":
        pass

    dataset_cfg.str = json.dumps(dataset_cfg.__dict__, indent=4)
    return dataset_cfg
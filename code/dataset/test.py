# This file is used to do very basic testing of the dataset class.

from .dataset import AVDataset, TVDataset, TVADataset

# test AVDataset
def test_avdataset():
    args = type("args", (object,), {})()
    args.dataset = "CREMAD"
    
    dataset = AVDataset(args, "train")
    assert dataset is not None
    assert len(dataset) == 5962
    # print(f"Dataset length: {len(dataset)}")
    data = dataset[0]
    assert dataset[0] is not None
    assert len(data) == 4
    
    dataset = AVDataset(args, "test")
    assert dataset is not None
    assert len(dataset) == 742
    
    dataset = AVDataset(args, "val")
    assert dataset is not None
    assert len(dataset) == 742
    
    print("Passed ", test_avdataset.__name__)

# test TVDataset
def test_tv_dataset():
    args = type("args", (object,), {})()
    args.dataset = "MVSA"
    
    dataset = TVDataset(args, "train")
    assert dataset is not None
    assert len(dataset) == 3552
    data = dataset[0]
    assert dataset[0] is not None
    assert len(data) == 5
    
    dataset = TVDataset(args, "test")
    assert dataset is not None
    assert len(dataset) == 481
    
    dataset = TVDataset(args, "val")
    assert dataset is not None
    assert len(dataset) == 478
    
    print("Passed ", test_tv_dataset.__name__)

# test TVADataset
def test_tvav_dataset():
    args = type("args", (object,), {})()
    args.dataset = "URFUNNY"
    
    dataset = TVADataset(args, "train")
    assert dataset is not None
    assert len(dataset) == 8132
    data = dataset[0]
    assert dataset[0] is not None
    assert len(data) == 6
    
    dataset = TVADataset(args, "test")
    assert dataset is not None
    assert len(dataset) == 1018
    
    dataset = TVADataset(args, "val")
    assert dataset is not None
    assert len(dataset) == 1016
    
    print("Passed ", test_tvav_dataset.__name__)
    
test_avdataset()   
test_tv_dataset()
test_tvav_dataset()
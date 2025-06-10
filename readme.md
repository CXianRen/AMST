This repository contains the official implementation of our paper: AMST: Alternating Multimodal Skip Training.

The code has been refactored from the initial version, with efforts made to keep it as clean and readable as possible.
We strive to maintain consistency with the version used in our paper.

If you encounter any issues or have suggestions, your feedback is highly appreciated.
We will do our best to maintain and improve this repository—although, as you might understand, it can sometimes be challenging.


# Env
Most time it should work:
```sh
pip install -r requirement.txt
```

# Dataset
| Dataset   | V | A | T | Link                                         | Example         |
|-----------|---|---|---|----------------------------------------------|-----------------|
| CREMAD    | √ | √ | × | [CREMAD](./Doc/dataset/CREMAD.md)           | <img src="Doc/dataset/imgs/example_cremad.png" alt="CREMAD example" width="100" height="50">|
| AVE       | √ | √ | × | [AVE](./Doc/dataset/AVE.md)                 | <img src="Doc/dataset/imgs/example_ave.png" alt="AVE example" width="100" height="50"> |
| MVSA      | √ | × | √ | [MVSA](./Doc/dataset/MVSA.md)               | <img src="Doc/dataset/imgs/example_mvsa.jpg" alt="mvsa example" width="100" height="50"> |
| IEMOCAP   | √ | √ | √ | [IEMOCAP](./Doc/dataset/IEMOCAP.md)         | <img src="Doc/dataset/imgs/example_iemo.jpg" alt="IEMO example" width="100" height="50"> |
| UR-FUNNY  | √ | √ | √ | [UR-FUNNY](./Doc/dataset/UR-FUNNY.md)       | <img src="Doc/dataset/imgs/example_uf.jpg" alt="UF example" width="100" height="50"> |

* For CREMAD dataset, in previous work, they didn't have a validating set, so we split it into 80% training, 10% validating, 10% testing.  

* Similar handling for all other dataset without a validating set.

* For AVE dataset, we used the original splited train valid and test set. But we notice there some samples have more than one labels. 


# How to run
Most running srcripts you can find it under the "./Jobs/xx.sh".

## Test dataset
You can run a simple test for debugging when your data is ready.
```sh
python3 -m dataset.test
```


## Checkpoints
we have uploaded parts of our model here [ckpt](https://drive.google.com/drive/folders/1x9TER3mc1sMgcALp7x_ooK65IjRHOaHN?usp=sharing). You can download and run evaluated directly with:

```sh
# single modal model
python3 -m evaluate.eval_sm --dataset CREMAD --modalit visual --model_path `[YOUR CHECKPOINTS PATH]`
```

```sh
# multimodal model
python3 -m evaluate.eval_mm \
  --model_path `[YOUR CHECKPOINTS PATH]`
```
# Learning to Plan with Uncertain Topological Maps

This repository contains code for paper "Learning to Plan with Uncertain Topological Maps"( https://arxiv.org/abs/2007.05270) ECCV 2020 (Spotlight)

# Contents
* [Requirements]
* [Dataset]
* [Training]
* [Pretrained model]
* [Citation]



# Requirements:
pip install -R requirements.txt



# Dataset
Download the dataset from the following site (TODO) , note it is 15GB in size.
mkdir data 
cd data
wget xxxx.gz (TODO)


# Training

Please use the following for training:

```
python train.py --data_path data/graph_data3_distance_weights/train --hidden_size 256 --batch_size 32 --max_grad_norm 2 --weight_decay 0.0001 --lr 0.001 --schedule 120 --n_steps 6 --data_limit 74000 --use_weights --normalize --gru_depth 2 --save_model --bound_update --new_bound_net --store_hidden --use_probs --use_features

```

# Pretrained model
The pretrained model can be found in models/best_model.pth



# Citation


## Citation
If you find this useful, consider citing the following:
```
@inproceedings{beeching2020learntoplan,
  title={Learning to plan with uncertain topological maps.
  },
  author={Beeching, Edward and Dibangoye, Jilles and 
          Simonin, Olivier and Wolf, Christian}
  booktitle={European Conference on Computer Vision},
  year={2020}}
```




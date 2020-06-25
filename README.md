# RippleNet
A Tensorflow 2.x implementation of RippleNet

This repository is the implementation of RippleNet ([arXiv](https://arxiv.org/abs/1803.03467)):
> RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems  
Hongwei Wang, Fuzheng Zhang, Jialin Wang, Miao Zhao, Wenjie Li, Xing Xie, Minyi Guo  
The 27th ACM International Conference on Information and Knowledge Management (CIKM 2018)

![](framework.jpg)

RippleNet is a deep end-to-end model that naturally incorporates the knowledge graph into recommender systems.
Ripple Network overcomes the limitations of existing embedding-based and path-based KG-aware recommendation methods by introducing preference propagation, which automatically propagates users' potential preferences and explores their hierarchical interests in the KG.

### Files in the folder

- `data/`
  - `book/`
    - `BX-Book-Ratings.csv`: raw rating file of Book-Crossing dataset;
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
  - `movie/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg_part1.txt` and `kg_part2.txt`: knowledge graph file;
    - `ratrings.dat`: raw rating file of MovieLens-1M;
- `model/`: implementations of RippleNet.
- `tools/`: load data and model metrics.

### Setup your Python runtime

```
$ python 3.7 -m venv venv
$ source venv/bin/active
$ pip install -r requirements.txt
```

### Download dataset

**please download the `data` file from TAI platform**

- first of all, login TAI platform file system [TAI](https://ai.tezign.com/web-filesystem)
- and then download [data.zip](https://ai.tezign.com/web-filesystem?path=/data/Models/RippleNet-dataset/data.zip) to your project root path like above.
- finally, use command `$ unzip data.zip` to unzip the data.zip file.

### Required packages
The code has been tested running under Python 3.7, with the following packages installed (along with their dependencies):
- tensorflow-gpu == 2.2.0
- numpy == 1.18.5


### Running the code
```
$ cd RippleNet
$ mkdir logs
$ python preprocess.py --dataset movie (or --dataset book)
$ python main.py --dataset movie (note: use -h to check optional arguments)
```

### View tensorboard
```
$ cd RippleNet
$ tensorboard --logdir=logs/movie_%date% (or --logdir=logs/book_%date%)
```
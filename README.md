# About

This repository is a Tensorflow 2 implementation of RippleNet ([arXiv](https://arxiv.org/abs/1803.03467)):
> **RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems**, Hongwei Wang, Fuzheng Zhang, Jialin Wang, Miao Zhao, Wenjie Li, Xing Xie, Minyi Guo, The 27th ACM International Conference on Information and Knowledge Management (CIKM 2018)

![](framework.jpg)

>RippleNet is a deep end-to-end model that naturally incorporates the knowledge graph into recommender systems.
Ripple Network overcomes the limitations of existing embedding-based and path-based KG-aware recommendation methods by introducing preference propagation, which automatically propagates users' potential preferences and explores their hierarchical interests in the KG.

You can find other implementations below:

- [Authors' official Tensorflow 1.x implementation of RippleNet by @hwwang55](https://github.com/hwwang55/RippleNet)
- [A Tensorflow 2 implementation of RippleNet by @SSSxCCC](https://github.com/SSSxCCC/Recommender-System)
- [A PyTorch implementation of RippleNet by @qibinc](https://github.com/qibinc/RippleNet-PyTorch)

## Environment Setup

Tested using Python 3.7 and Tensorflow 2.2.0. You should setup the following virtual environment and install the required packages:

```
python3 -m venv venv
source venv/bin/active
pip install -r requirements.txt
```
## Data

Unzip the `data.zip` file to the project root and your folder structure should look like the following (note that `/data` folder is gitignored):

```
.
├── LICENSE
├── README.md
├── data
│   ├── book
│   │   ├── book_ratings.csv
│   │   ├── item_index2entity_id_rehashed.txt
│   │   └── kg_rehashed.txt
│   └── movie
│       ├── item_index2entity_id_rehashed.txt
│       ├── kg_part1_rehashed.txt
│       ├── kg_part2_rehashed.txt
│       └── movie_ratings.dat
├── framework.jpg
├── main.py
├── model
│   ├── __init__.py
│   ├── layers.py
│   ├── model.py
│   └── ripple_net.py
├── preprocess.py
├── requirements.txt
└── tools
    ├── __init__.py
    ├── load_data.py
    └── metrics.py

```

- `model/`: implementation of RippleNet
- `tools/`: data loader and model metrics
- `data/book/`
  - `book_ratings.csv`: raw rating file of Book-Crossing dataset
  - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG
  - `kg.txt`: book knowledge graph file
- `data/movie/` 
  - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG
  - `kg_part1.txt` and `kg_part2.txt`: movie knowledge graph files
  - `movie_ratings.dat`: raw rating file of MovieLens-1M;


## Run

for the movie dataset (for the book dataset, replace movie with book in the commands):

- run `python preprocess.py --dataset movie` - this will generate two new files `kg_final.txt` and `ratings_final.txt`
- run `python main.py --dataset movie` - this will start the training, create a `logs` folder, and do a final evaluation

By default, the model is trained using 10 epochs, which takes about 25 minutes on a MacBook Pro (3.1 GHz Dual-Core Intel Core i5 with 8G RAM). One sample evaluation result is as follows:

```
evaluate model ...
148/148 [==============================] - 13s 88ms/step - loss: 0.3811 - binary_accuracy: 0.8464 - auc: 0.9221 - f1: 0.8486 - precision: 0.8399 - recall: 0.8577
- loss: 0.3810572326183319 - binary_accuracy: 0.8464029431343079 - auc: 0.9220795631408691 - f1: 0.848639190196991 - precision: 0.8399457931518555 - recall: 0.8577455878257751
```


You can use Tensorboard to check the training results (in real time or after training) by running `tensorboard --logdir=logs` and then point your browser to http://localhost:6006/

<img width="848" alt="Screen Shot 2020-06-26 at 11 18 30 AM" src="https://user-images.githubusercontent.com/595772/85873541-48f63700-b79f-11ea-9b29-2b9d4bf9a984.png">
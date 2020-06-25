from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from tools.load_data import LoadData
import numpy as np
import datetime
import math


class BuildModel:
    def __init__(self, args):
        self.args = args
        self._parse_args()
        self.data_info = LoadData(args)
        self.train_data, self.test_data, self.n_entity, self.n_relation, self.ripple_set = self.data_info.load_data()
        self.model = self.build_model()

    def _parse_args(self):
        self.batch_size = self.args.batch_size
        self.epochs = self.args.n_epoch
        self.patience = self.args.patience
        self.dim = self.args.dim
        self.n_hop = self.args.n_hop
        self.kge_weight = self.args.kge_weight
        self.l2_weight = self.args.l2_weight
        self.lr = self.args.lr
        self.n_memory = self.args.n_memory
        self.item_update_mode = self.args.item_update_mode
        self.using_all_hops = self.args.using_all_hops
        self.save_path = self.args.base_path + "/data/" + self.args.dataset
        self.save_path += "/ripple_net_{}_model.h5".format(self.args.dataset)
        current_time = datetime.datetime.now().strftime(('%Y%m%d-%H%M%S'))
        self.log_path = self.args.base_path + "/logs/{}_{}".format(self.args.dataset, current_time)

    def step_decay(self, epoch):
        # learning rate step decay
        initial_l_rate = self.lr
        drop = 0.5
        epochs_drop = 10.0
        l_rate = initial_l_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        print("learning_rate", l_rate)
        return l_rate

    def build_model(self):
        pass

    def data_parse(self, data):
        # build X, y from data
        np.random.shuffle(data)
        items = data[:, 1]
        labels = data[:, 2]
        memories_h = list(range(self.n_hop))
        memories_r = list(range(self.n_hop))
        memories_t = list(range(self.n_hop))
        for hop in range(self.n_hop):
            memories_h[hop] = np.array([self.ripple_set[user][hop][0] for user in data[:, 0]])
            memories_r[hop] = np.array([self.ripple_set[user][hop][1] for user in data[:, 0]])
            memories_t[hop] = np.array([self.ripple_set[user][hop][2] for user in data[:, 0]])
        return [items, labels] + memories_h + memories_r + memories_t, labels

    def train(self):
        print("train model ...")
        self.model.summary()
        X, y = self.data_parse(self.train_data)
        tensorboard = TensorBoard(log_dir=self.log_path, histogram_freq=1)
        early_stopper = EarlyStopping(patience=self.patience, verbose=1)
        model_checkpoint = ModelCheckpoint(self.save_path, verbose=1, save_best_only=True)
        learning_rate_scheduler = LearningRateScheduler(self.step_decay)
        self.model.fit(x=X,
                       y=y,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_split=0.2,
                       callbacks=[early_stopper, model_checkpoint, learning_rate_scheduler, tensorboard])

    def evaluate(self):
        model = self.build_model()
        model.load_weights(self.save_path)
        print("evaluate model ...")
        X, y = self.data_parse(self.test_data)
        score = model.evaluate(X, y, batch_size=self.batch_size)
        print("- loss: {} "
              "- binary_accuracy: {} "
              "- auc: {} "
              "- f1: {} "
              "- precision: {} "
              "- recall: {}".format(*score))

    def predict(self):
        model = self.build_model()
        model.load_weights(self.save_path)
        X, y = self.data_parse(self.test_data)
        pred = model.predict(X, batch_size=self.batch_size)
        result = [1 if x > 0.5 else 0 for x in pred]
        return result

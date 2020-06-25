from tensorflow.keras.layers import Embedding, Input, Dense, Softmax, Activation, Lambda
from model.layers import Squeeze, ExpandDims, Embedding2D
from tensorflow.keras.losses import binary_crossentropy
from tools.metrics import auc, f1, precision, recall
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from model.model import BuildModel
from tensorflow import keras
import tensorflow as tf


class RippleNet(BuildModel):

    def update_item_embedding(self, item_embeddings, o, l2):
        # transformation matrix for updating item embeddings at the end of each hop
        transform_matrix = Dense(self.dim, use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=l2)
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = transform_matrix(o)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = transform_matrix(item_embeddings + o)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings

    def build_model(self):
        # Input Tensor
        item_inputs = Input(shape=(), name="items", dtype=tf.int32)
        label_inputs = Input(shape=(), name="labels", dtype=tf.float32)
        h_inputs = []
        r_inputs = []
        t_inputs = []

        for hop in range(self.n_hop):
            h_inputs.append(Input(shape=(self.n_memory,), name="h_inputs_{}".format(hop), dtype=tf.int32))
            r_inputs.append(Input(shape=(self.n_memory,), name="r_inputs_{}".format(hop), dtype=tf.int32))
            t_inputs.append(Input(shape=(self.n_memory,), name="t_inputs_{}".format(hop), dtype=tf.int32))

        # Matmul layer
        matmul = Lambda(lambda x: tf.matmul(x[0], x[1]))

        # Embedding layer
        l2 = keras.regularizers.l2(self.l2_weight)
        entity_embedding = Embedding(self.n_entity,
                                     self.dim,
                                     embeddings_initializer='glorot_uniform',
                                     embeddings_regularizer=l2,
                                     name="entity_embedding")
        relation_embedding = Embedding2D(self.n_relation,
                                         self.dim,
                                         self.dim,
                                         embeddings_initializer='glorot_uniform',
                                         embeddings_regularizer=l2,
                                         name="relation_embedding")

        # item and ripple embedding
        # [batch size, dim]
        item_embeddings = entity_embedding(item_inputs)
        h_embeddings = []
        r_embeddings = []
        t_embeddings = []
        for hop in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_embeddings.append(entity_embedding(h_inputs[hop]))

            # [batch size, n_memory, dim, dim]
            r_embeddings.append(relation_embedding(r_inputs[hop]))

            # [batch size, n_memory, dim]
            t_embeddings.append(entity_embedding(t_inputs[hop]))

        # update item embedding
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            reshape_h = ExpandDims(3)(h_embeddings[hop])

            # [batch_size, n_memory, dim]
            Rh = matmul([r_embeddings[hop], reshape_h])
            Rh = Squeeze(3)(Rh)

            # [batch_size, dim, 1]
            v = ExpandDims(2)(item_embeddings)

            # [batch_size, n_memory]
            probs = Squeeze(2)(matmul([Rh, v]))

            # [batch_size, n_memory]
            probs_normalized = Softmax()(probs)

            # [batch_size, n_memory, 1]
            probs_normalized = ExpandDims(2)(probs_normalized)

            # [batch_size, dim]
            o = keras.backend.sum(t_embeddings[hop] * probs_normalized, axis=1)

            item_embeddings = self.update_item_embedding(item_embeddings, o, l2)
            o_list.append(o)

        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]

        # Output
        scores = Squeeze()(keras.backend.sum(item_embeddings * y, axis=1))
        scores_normalized = Activation('sigmoid', name='score')(scores)

        # Model
        model = Model(inputs=[item_inputs, label_inputs] + h_inputs + r_inputs + t_inputs, outputs=scores_normalized)

        # Loss
        base_loss = binary_crossentropy(label_inputs, scores_normalized)  # base loss

        kge_loss = 0  # kg loss
        for hop in range(self.n_hop):
            h_expanded = ExpandDims(2)(h_embeddings[hop])
            t_expanded = ExpandDims(3)(t_embeddings[hop])
            hRt = Squeeze()(h_expanded @ r_embeddings[hop] @ t_expanded)
            kge_loss += keras.backend.mean(Activation('sigmoid')(hRt))

        l2_loss = 0  # l2 loss
        for hop in range(self.n_hop):
            l2_loss += keras.backend.sum(keras.backend.square(h_embeddings[hop]))
            l2_loss += keras.backend.sum(keras.backend.square(r_embeddings[hop]))
            l2_loss += keras.backend.sum(keras.backend.square(t_embeddings[hop]))

        model.add_loss(base_loss)
        model.add_loss(self.l2_weight * l2_loss)
        model.add_loss(self.kge_weight * -kge_loss)
        model.compile(optimizer=Adam(self.lr), metrics=[binary_accuracy, auc, f1, precision, recall])
        return model

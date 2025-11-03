import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np

import tensorflow as tf

# @tf.function
# def llp_loss(y_true, y_pred):
#     """
#     y_true: (batch, 2) -> [pi_s, sigma_pi_s]
#     y_pred: (batch, 1) -> predicted probabilities for signal
#     """
#     pi_s   = y_true[:, 0]
#     sigma  = y_true[:, 1]

#     # Mean predicted signal probability within batch
#     q_mean = tf.reduce_mean(y_pred)

#     # Mean π_s, σ per batch
#     pi_mean = tf.reduce_mean(pi_s)
#     sigma_mean = tf.reduce_mean(sigma)

#     # Weighted squared deviation (normalized by σ)
#     diff = (q_mean - pi_mean) / (sigma_mean + 1e-6)
#     loss = tf.square(diff)
#     return loss

# @tf.function
# def llp_loss(y_true, y_pred):
#     pi_s   = y_true[:, 0]
#     sigma  = y_true[:, 1]
#     bin_id = tf.cast(y_true[:, 2], tf.int32)

#     num_bins = tf.reduce_max(bin_id) + 1

#     # mean prediction per bin
#     q_mean = tf.math.unsorted_segment_mean(tf.squeeze(y_pred, -1), bin_id, num_bins)
#     pi_mean = tf.math.unsorted_segment_mean(pi_s, bin_id, num_bins)
#     sigma_mean = tf.math.unsorted_segment_mean(sigma, bin_id, num_bins)

#     sigma_eff = sigma_mean + 0.05
#     per_bin = tf.square((q_mean - pi_mean) / (sigma_eff + 1e-6))
#     return tf.reduce_mean(per_bin)

@tf.function
def llp_loss(y_true, y_pred):
    pi_s   = y_true[:, 0]
    sigma  = y_true[:, 1]
    bin_id = tf.cast(y_true[:, 2], tf.int32)

    num_bins = tf.reduce_max(bin_id) + 1
    q_mean = tf.math.unsorted_segment_mean(tf.squeeze(y_pred, -1), bin_id, num_bins)
    pi_mean = tf.math.unsorted_segment_mean(pi_s, bin_id, num_bins)
    sigma_mean = tf.math.unsorted_segment_mean(sigma, bin_id, num_bins)
    sigma_eff = sigma_mean + 0.05
    per_bin = tf.square((q_mean - pi_mean) / (sigma_eff + 1e-6))
    llp_term = tf.reduce_mean(per_bin)

    # Global normalization + entropy
    q_global = tf.reduce_mean(y_pred)
    pi_global = tf.reduce_mean(pi_s)
    norm_term = tf.square(q_global - pi_global)
    p = tf.clip_by_value(y_pred, 1e-6, 1-1e-6)
    entropy_reg = -tf.reduce_mean(p * tf.math.log(p) + (1 - p) * tf.math.log(1 - p))

    # Tunable hyperparameters
    return llp_term + 0.2 * norm_term - 0.02 * entropy_reg




class SignalNN:
    def __init__(self, input_shape, pi_s=None):
        self.model = self.build_model(input_shape, pi_s)

    def build_model(self, input_shape, pi_s):
        mean_pi = np.mean(pi_s)  # global mean prior
        bias_init = np.log(mean_pi / (1 - mean_pi))
        model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        # layers.Dropout(0.2),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        # layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid',
                     bias_initializer=tf.keras.initializers.Constant(bias_init))
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=llp_loss)
        return model

    def train(self, dataset, epochs=20, steps_per_epoch=200):
        lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6)
        early_stopper = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        self.model.fit(dataset,
                       epochs=epochs,
                       steps_per_epoch=steps_per_epoch,
                       callbacks=[lr_reducer, early_stopper],
                       verbose=1)

    def predict(self, X):
        return self.model.predict(X)

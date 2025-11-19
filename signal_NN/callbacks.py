import numpy as np
from tensorflow import keras

class TrainingLogger(keras.callbacks.Callback):
    def __init__(self, X_val, MM_val, pi_s, bin_edges, clf_ref):
        super().__init__()
        self.X_val = X_val
        self.MM_val = MM_val
        self.pi_s = pi_s
        self.bin_edges = bin_edges
        self.clf_ref = clf_ref

        self.history = {
            "loss": [],
            "val_loss": [],
            "acc": [],
            "val_acc": [],
            "mean_score": [],
            "std_score": [],
            "mean_Q": [],
            "std_Q": []
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # store loss and accuracy
        self.history["loss"].append(logs.get("loss", np.nan))
        self.history["val_loss"].append(logs.get("val_loss", np.nan))
        self.history["acc"].append(logs.get("accuracy", np.nan))
        self.history["val_acc"].append(logs.get("val_accuracy", np.nan))

        # compute scores on validation sample
        scores = self.clf_ref.predict_scores(self.X_val)
        self.history["mean_score"].append(np.mean(scores))
        self.history["std_score"].append(np.std(scores))

        # compute ranked Q on validation sample
        Q_val = self.clf_ref.ranked_Q(scores, self.MM_val, self.pi_s)
        self.history["mean_Q"].append(np.mean(Q_val))
        self.history["std_Q"].append(np.std(Q_val))
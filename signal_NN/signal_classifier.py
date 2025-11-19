import numpy as np
import tensorflow as tf
from tensorflow import keras
from .callbacks import TrainingLogger


class SidebandClassifier:
    def __init__(self, bin_edges, SB_low=(0.65,0.80), SB_high=(1.10,1.25)):
        self.bin_edges = bin_edges
        self.SB_low = SB_low
        self.SB_high = SB_high
        self.model = None
        self.train_history = None

    # ---------------------------------------------------
    # region masks
    # ---------------------------------------------------
    def build_region_masks(self, MM):
        SB_low = (MM >= self.SB_low[0]) & (MM < self.SB_low[1])
        SB_high = (MM >= self.SB_high[0]) & (MM < self.SB_high[1])
        sideband_mask = SB_low | SB_high
        mixture_mask = ~sideband_mask
        return sideband_mask, mixture_mask

    # ---------------------------------------------------
    # Neural network architecture
    # ---------------------------------------------------
    def build_classifier(self, input_dim):
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    # ---------------------------------------------------
    # Main training function
    # ---------------------------------------------------
    def train(self, X, MM, verbose=1, X_val=None, MM_val=None, pi_s=None, **kwargs):
        sideband_mask, mixture_mask = self.build_region_masks(MM)

        X_bg  = X[sideband_mask][:,1:]
        X_mix = X[mixture_mask][:,1:]

        y_bg  = np.zeros(len(X_bg))
        y_mix = np.ones(len(X_mix))

        X_cls = np.vstack([X_bg, X_mix])
        y_cls = np.concatenate([y_bg, y_mix])

        model = self.build_classifier(input_dim=X_cls.shape[1])

        callbacks = []
        if X_val is not None:
            logger = TrainingLogger(X_val, MM_val, pi_s, self.bin_edges, self)
            callbacks.append(logger)

        fit_args = dict(
            epochs=50,
            batch_size=256,
            verbose=verbose,
            callbacks=callbacks,
            shuffle=True,
        )

        fit_args.update(kwargs)

        hist = model.fit(
            X_cls, y_cls,
            validation_split=0.2 if X_val is None else 0.0,
            **fit_args,
        )

        self.model = model
        self.train_history = logger.history if X_val is not None else hist.history
        return model

    # ---------------------------------------------------
    # Predict classifier scores
    # ---------------------------------------------------
    def predict_scores(self, X):
        X_features = X[:,1:]     # drop MM
        scores = self.model.predict(X_features, batch_size=5000, verbose=0).flatten()
        return scores

    # ---------------------------------------------------
    # ranked-Q method
    # ---------------------------------------------------
    def ranked_Q(self, scores, MM, pi_s):
        bin_id = np.digitize(MM, self.bin_edges) - 1
        num_bins = len(self.bin_edges)-1
        Q = np.zeros_like(scores)

        for b in range(num_bins):
            idx = np.where(bin_id == b)[0]
            if len(idx) == 0:
                continue
            Nb = len(idx)
            Ns = int(round(pi_s[b] * Nb))
            if Ns <= 0:
                continue
            order = idx[np.argsort(scores[idx])[::-1]]
            Q[order[:Ns]] = 1.0
        return Q

    def ranked_Q_q2(self, scores, MM, Q2, pi_s_all, pi_s_all_unc, Q2_cut):
        """
        scores : classifier scores
        MM     : invariant mass array
        Q2     : Q2 array
        pi_s_all : dict keyed by (q2_min, q2_max) -> pi_s array (per MM bin)
        Q2_cut: array of Q2 bin edges (length n_Q2_bins+1)
        """
        mm_bin = np.digitize(MM, self.bin_edges) - 1
        mm_bin = np.clip(mm_bin, 0, len(self.bin_edges)-2)

        q2_bin = np.digitize(Q2, Q2_cut) - 1
        q2_bin = np.clip(q2_bin, 0, len(Q2_cut)-2)

        Q_mid = np.zeros_like(scores, dtype=float)
        Q_low = np.zeros_like(scores, dtype=float)
        Q_high = np.zeros_like(scores, dtype=float)

        for k in range(len(Q2_cut)-1):
            q2_min = Q2_cut[k]
            q2_max = Q2_cut[k+1]

            mask_q2 = (q2_bin == k)
            if not np.any(mask_q2):
                continue

            # π_s array for this Q2 bin
            pi_s_k = pi_s_all[(q2_min, q2_max)]
            pi_s_unc_k = pi_s_all_unc[(q2_min, q2_max)]

            for b in range(len(self.bin_edges)-1):
                idx = np.where(mask_q2 & (mm_bin == b))[0]
                if len(idx) == 0:
                    continue

                # for each (Q2,MM) bin:
                Nb = len(idx)
                pi  = pi_s_k[b]
                dpi = pi_s_unc_k[b]     # THIS is new (uncertainty band)

                # compute allowed number of signal events
                Ns_mid  = int(round(pi * Nb))
                Ns_low  = int(round((pi - dpi) * Nb))
                Ns_high = int(round((pi + dpi) * Nb))

                # clamp
                Ns_low  = max(0, Ns_low)
                Ns_high = min(Nb, Ns_high)

                # ranking order: sorted by score
                order = idx[np.argsort(scores[idx])[::-1]]

                # fill bands:
                Q_mid[order[:Ns_mid]]   = 1.0
                Q_low[order[:Ns_low]]   = 1.0
                Q_high[order[:Ns_high]] = 1.0

        return {"Q_mid": Q_mid, "Q_low": Q_low, "Q_high": Q_high}

    # ---------------------------------------------------
    # predict all Q for a dataset
    # ---------------------------------------------------
    def predict_Q_all(self, X, MM, pi_s):
        scores = self.predict_scores(X)
        Q = self.ranked_Q(scores, MM, pi_s)
        return Q, scores

    def predict_Q_all_q2(self, X, MM, Q2, pi_s_all, pi_s_all_unc, Q2_cut):
        scores = self.predict_scores(X)
        Q = self.ranked_Q_q2(scores, MM, Q2, pi_s_all, pi_s_all_unc, Q2_cut)
        return Q, scores
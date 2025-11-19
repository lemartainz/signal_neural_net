import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class QFactorDiagnostics:
    def __init__(self, classifier, X, MM, pi_s, pi_s_unc, bin_edges):
        self.clf = classifier
        self.X = X
        self.MM = MM
        self.pi_s = pi_s
        self.pi_s_unc = pi_s_unc
        self.bin_edges = bin_edges

    # ---------------------------------------------------
    # run ensemble of classifiers
    # ---------------------------------------------------
    def run_ensemble(self, X_train, MM_train, K=10, verbose=0):
        Q_list = []

        for k in range(K):
            print(f"Training ensemble member {k+1}/{K}...")
            self.clf.train(X_train, MM_train, verbose=verbose)
            Q, _ = self.clf.predict_Q_all(self.X, self.MM, self.pi_s)
            Q_list.append(Q)

        self.Q_ensemble = np.array(Q_list)  # shape (K, N)
        self.Q_mean = np.mean(self.Q_ensemble, axis=0)
        self.Q_std  = np.std(self.Q_ensemble, axis=0)

        return self.Q_mean, self.Q_std

    # ---------------------------------------------------
    # Q statistical uncertainty
    # ---------------------------------------------------
    def Q_stat_unc(self, MM_range=None):
        if MM_range is None:
            mask = np.ones_like(self.MM, dtype=bool)
        else:
            lo, hi = MM_range
            mask = (self.MM >= lo) & (self.MM < hi)

        Q_sel = self.Q_mean[mask]
        Y = np.sum(Q_sel)
        var = np.sum(Q_sel * (1.0 - Q_sel))
        return Y, np.sqrt(var)

    # ---------------------------------------------------
    # model (ensemble) uncertainty on yield
    # ---------------------------------------------------
    def model_uncertainty(self, MM_range=None):
        if MM_range is None:
            mask = np.ones_like(self.MM, dtype=bool)
        else:
            lo, hi = MM_range
            mask = (self.MM >= lo) & (self.MM < hi)

        Q_std_sel = self.Q_std[mask]
        return np.sqrt(np.sum(Q_std_sel**2))

    # ---------------------------------------------------
    # prior uncertainty
    # ---------------------------------------------------
    def prior_uncertainty(self):
        bin_id = np.digitize(self.MM, self.bin_edges) - 1
        num_bins = len(self.bin_edges)-1
        N_b = np.array([np.sum(bin_id == b) for b in range(num_bins)])

        var = np.sum((N_b * self.pi_s_unc)**2)
        return np.sqrt(var)

    # ---------------------------------------------------
    # full uncertainty combo
    # ---------------------------------------------------
    def total_uncertainty(self, MM_range=None):
        Y, sigma_stat = self.Q_stat_unc(MM_range)
        sigma_model   = self.model_uncertainty(MM_range)
        sigma_prior   = self.prior_uncertainty()

        sigma_tot = np.sqrt(sigma_stat**2 + sigma_model**2 + sigma_prior**2)
        return Y, sigma_tot, sigma_stat, sigma_model, sigma_prior

    # ---------------------------------------------------
    # per-bin table
    # ---------------------------------------------------
    def yield_table(self, Q_true=None, y_true=None):
        bin_id = np.digitize(self.MM, self.bin_edges) - 1
        num_bins = len(self.bin_edges)-1

        true_yield = np.zeros(num_bins)
        est_yield  = np.zeros(num_bins)
        rel_bias   = np.zeros(num_bins)

        for b in range(num_bins):
            mask = (bin_id == b)
            est_yield[b] = self.Q_mean[mask].sum()

            if Q_true is not None:
                true_yield[b] = Q_true[mask].sum()
                if true_yield[b] > 0:
                    rel_bias[b] = (est_yield[b] - true_yield[b]) / true_yield[b]
                else:
                    rel_bias[b] = np.nan
            else:
                true_yield[b] = np.nan
                rel_bias[b] = np.nan

        centers = 0.5*(self.bin_edges[:-1] + self.bin_edges[1:])

        df = pd.DataFrame({
            "MM_center": centers,
            "True_Yield": true_yield,
            "Q_Mean_Yield": est_yield,
            "Rel_Bias": rel_bias
        })
        return df
    
    # ---------------------------------------------------
    # Compute ROC + AUC using sklearn
    # ---------------------------------------------------
    def compute_roc(self, y_true, scores):
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        auc_val = auc(fpr, tpr)
        return fpr, tpr, thresholds, auc_val

    # ---------------------------------------------------
    # Plot ROC curve
    # ---------------------------------------------------
    def plot_roc(self, y_true, scores):
        fpr, tpr, thresholds, auc_val = self.compute_roc(y_true, scores)

        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.4f}")
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return auc_val
    
    # ---------------------------------------------------
    # Plot feature distributions
    # ---------------------------------------------------
    def plot_feature_distributions(self, X, y_true, Q_pred):
        n_features = X.shape[1]
        ncols = 3
        nrows = int(np.ceil(n_features / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
        axes = axes.flatten()

        for i in range(n_features):
            ax = axes[i]
            fname = f"var_{i}"

            # true signal
            ax.hist(X[y_true==1, i], bins=60, range=(X[y_true==1, i].min(), X[y_true==1, i].max()), histtype='step',
                    color='red', label='True Signal')

            # predicted signal
            ax.hist(X[:, i], bins=60, range=(X[y_true==1, i].min(), X[y_true==1, i].max()),
                    weights=Q_pred, histtype='step',
                    color='blue', label='Predicted Signal')

        #     # true background
        #     ax.hist(X[y_true==0, i], bins=60, density=True, histtype='step',
        #             color='orange', label='True Background')

        #     # predicted background
        #     ax.hist(X[:, i], bins=60, density=True,
        #             weights=(1-Q_pred), histtype='step',
        #             color='green', label='Predicted Background')

            ax.set_title(fname)
            ax.legend()

        # remove unused axes
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        return fig

    
    def plot_loss(self, clf):
        h = clf.train_history
        plt.figure(figsize=(7,4))
        plt.plot(h["loss"], label="loss")
        if "val_loss" in h:
            plt.plot(h["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_Q_stats(self, clf):
        h = clf.train_history
        plt.figure(figsize=(12,4))

        plt.subplot(1,2,1)
        plt.plot(h["mean_Q"])
        plt.title("mean Q vs epoch")
        plt.xlabel("epoch")

        plt.subplot(1,2,2)
        plt.plot(h["std_Q"])
        plt.title("std Q vs epoch")
        plt.xlabel("epoch")

        plt.tight_layout()
        plt.show()

    def plot_score_stats(self, clf):
        h = clf.train_history
        plt.figure(figsize=(12,4))

        plt.subplot(1,2,1)
        plt.plot(h["mean_score"])
        plt.title("mean score vs epoch")
        plt.xlabel("epoch")

        plt.subplot(1,2,2)
        plt.plot(h["std_score"])
        plt.title("std score vs epoch")
        plt.xlabel("epoch")

        plt.tight_layout()
        plt.show()
import numpy as np
from sklearn.model_selection import train_test_split

def generate_synthetic_data(num_events:float=1_000_000, n_features:int=6, signal_fraction:float=0.5, random_state:int=42):
    np.random.seed(random_state)

    num_signal = int(num_events * signal_fraction)
    num_background = num_events - num_signal

    # Generate signal data
    x_signal = np.random.normal(loc=1.0, scale=0.05, size=(num_signal, n_features))

    # Generate background data
    x_background = np.random.normal(loc=2, scale=1, size=(num_background, n_features))

    # Combine signal and background data
    x = np.vstack([x_signal, x_background])
    y = np.concatenate((np.ones(num_signal), np.zeros(num_background)))

    idx = np.random.permutation(len(y))
    x, y = x[idx], y[idx]
    return train_test_split(x, y, test_size=0.2, random_state=random_state)


# ----------------------------------------------------
#    Correlated toy generator (no real physics inside)
#    Columns: [MM, p_mag, Q2, W, costheta]
# ----------------------------------------------------
def sample_costheta_signal(n, a=0.5):
    # forward-ish peaking: p(cosθ) ∝ 1 + a*cos^2θ
    u = np.random.uniform(0, 1, n)
    # approximate by importance mixing: draw many, keep those with weight
    m = max(3*n, 20000)
    z = np.random.uniform(-1, 1, m)
    w = 1 + a*z**2
    w /= w.max()
    accept = np.random.uniform(0, 1, m) < w
    out = z[accept][:n]
    if len(out) < n:  # fallback if rare
        out = np.pad(out, (0, n-len(out)), mode='edge')
    return out

def sample_costheta_background(n):
    # broader, near-flat with mild backward preference
    z = np.random.uniform(-1, 1, n)
    return 0.7*z - 0.1*np.sign(z) + 0.1*np.random.normal(size=n)

def generate_correlated_toy(num_events=100_000, signal_fraction=0.5, rng=42):
    rs = np.random.default_rng(rng)
    n_sig = int(num_events * signal_fraction)
    n_bkg = num_events - n_sig

    # --- Q2: skewed to lower values (common in DIS-ish kinematics, but toy) ---
    # Use Gamma to get a positive, right-skewed distribution.
    Q2_sig = rs.gamma(shape=2.0, scale=1.2, size=n_sig) + 0.5   # ~[0.5, ~8]
    Q2_bkg = rs.gamma(shape=1.6, scale=1.5, size=n_bkg) + 0.3   # broader

    # --- costheta: different shapes for S/B ---
    ct_sig = sample_costheta_signal(n_sig, a=0.6)
    ct_bkg = sample_costheta_background(n_bkg)
    ct_sig = np.clip(ct_sig, -1, 1)
    ct_bkg = np.clip(ct_bkg, -1, 1)

    # --- W correlated with Q2 (toy linear + noise; different slopes for S/B) ---
    # Keep W in a reasonable band [2.0, 3.0] (toy, not physical)
    W_sig = 2.10 + 0.12*np.sqrt(Q2_sig) + rs.normal(0, 0.04, n_sig)
    W_bkg = 2.00 + 0.18*np.sqrt(Q2_bkg) + rs.normal(0, 0.06, n_bkg)
    # W_sig = np.clip(W_sig, 1.9, 3.2)
    # W_bkg = np.clip(W_bkg, 1.9, 3.2)

    # --- p_mag correlated with W and costheta (toy functional forms) ---
    # Signal: tighter relation, less noise
    p_sig = 1.5*(W_sig - 2.0)*(1 - 0.25*ct_sig) + rs.normal(0, 0.05, n_sig)
    p_bkg = 1.6*(W_bkg - 2.0)*(1 - 0.10*ct_bkg) + rs.normal(0, 0.5, n_bkg)
    # p_sig = np.clip(p_sig, 0, None)
    # p_bkg = np.clip(p_bkg, 0, None)

    # --- MM is the main discriminant ---
    # Signal: narrow-ish peak near 0.938 with Q2-dependent resolution
    mu_sig = 0.938
    sigma_sig = 0.015 + 0.005*np.sqrt(Q2_sig)  # widens with Q2
    MM_sig = rs.normal(mu_sig, sigma_sig)
    # MM_sig = np.clip(MM_sig, 0.65, 1.25)

    # Background: smooth shape on [0.65, 1.25], slightly rising to higher MM
    # Mix a truncated normal + gentle exponential tilt to avoid a fake peak
    base_bkg = rs.normal(1.5, 0.4, size=n_bkg)
    # base_bkg = np.clip(base_bkg, 0.65, 1.25)
    # Add small tilt: MM_bkg = base + epsilon*(MM-1.0)
    eps = 0.05
    MM_bkg = base_bkg + eps*(base_bkg - 1.0)

    # Stack features
    X_sig = np.column_stack([MM_sig, p_sig, Q2_sig, W_sig, ct_sig])
    X_bkg = np.column_stack([MM_bkg, p_bkg, Q2_bkg, W_bkg, ct_bkg])

    X = np.vstack([X_sig, X_bkg])
    y = np.concatenate([np.ones(n_sig, dtype=int), np.zeros(n_bkg, dtype=int)])

    # Shuffle
    idx = rs.permutation(len(y))

    X = X[idx]
    y = y[idx]
    return train_test_split(X, y, test_size=0.2, random_state=rng)
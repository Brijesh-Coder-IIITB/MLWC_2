import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# set random seed to 67 as strictly required by the assignment
np.random.seed(67)

# data generation & part (a): feature extraction

# generate standard square 16-qam grid
snr_levels = [0, 5, 10, 15, 20, 25, 30]
coords = np.array([-3, -1, 1, 3])
I, Q = np.meshgrid(coords, coords)

# scale the complex points so the average symbol energy is exactly 1
symbols = (I.flatten() + 1j * Q.flatten()) / np.sqrt(10)

data = []
for snr in snr_levels:
    snr_linear = 10 ** (snr / 10)
    # calculate noise variance (signal power is 1)
    noise_var = 1.0 / snr_linear
    
    for sym_id, sym in enumerate(symbols):
        # generate 200 samples per constellation point for every snr value
        # split noise variance equally across real and imaginary components
        noise = np.sqrt(noise_var / 2) * (np.random.randn(200) + 1j * np.random.randn(200))
        rx = sym + noise
        
        for r_val in rx:
            rx_i = r_val.real
            rx_q = r_val.imag
            
            # calculate instantaneous amplitude (r) and phase (theta)
            r_amp = np.abs(r_val)
            theta = np.arctan2(rx_q, rx_i)
            
            data.append({
                'snr_db': snr,
                'sym_id': sym_id,
                'rx_I': rx_i,
                'rx_Q': rx_q,
                'r': r_amp,
                'theta': theta
            })

# compile all generated samples and features into a dataframe
df2 = pd.DataFrame(data)

# part (b): k-means evaluation at 25 db

# isolate the data subset corresponding to snr = 25 db
df_25 = df2[df2['snr_db'] == 25]
X_cartesian_25 = df_25[['rx_I', 'rx_Q']].values

inertias = []
silhouettes = []

# fit standard k-means using cartesian coordinates for k = 2 to 20
# use k-means++ initialization with n_init=10 and random_state=42
for k in range(2, 21):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42).fit(X_cartesian_25)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_cartesian_25, km.labels_))

# side-by-side subplots for inertia and silhouette coefficient vs k
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(range(2, 21), inertias, marker='o')
ax[0].set_title('Inertia vs K (25 dB)')
ax[0].set_xlabel('K')
ax[0].grid(True)

ax[1].plot(range(2, 21), silhouettes, marker='s')
ax[1].set_title('Silhouette Coefficient vs K (25 dB)')
ax[1].set_xlabel('K')
ax[1].grid(True)
plt.show()

# fit k-means specifically for k=16 at 25 db
km16 = KMeans(n_clusters=16, init='k-means++', n_init=10, random_state=42).fit(X_cartesian_25)

# generate a 2d scatter plot of the received samples colored by assigned cluster index
# overlay the 16 learned cluster centroids as distinct markers
plt.figure(figsize=(6, 6))
plt.scatter(X_cartesian_25[:, 0], X_cartesian_25[:, 1], c=km16.labels_, cmap='tab20', alpha=0.5, s=15)
plt.scatter(km16.cluster_centers_[:, 0], km16.cluster_centers_[:, 1], c='black', marker='X', s=100)
plt.title('16-QAM Cartesian Clusters at 25 dB')
plt.xlabel('rx_I')
plt.ylabel('rx_Q')
plt.grid(True)
plt.show()

# helper function to compute cluster purity against ground-truth symbol ids
def get_purity(y_true, y_pred):
    cm = pd.crosstab(y_true, y_pred)
    return np.sum(np.max(cm.values, axis=0)) / np.sum(cm.values) * 100

# evaluate cluster purity for the three distinct feature sets
# set 1: cartesian coordinates
purity_set1 = get_purity(df_25['sym_id'], km16.labels_)

# set 2: polar coordinates
X_polar_25 = df_25[['r', 'theta']].values
km_polar = KMeans(n_clusters=16, init='k-means++', n_init=10, random_state=42).fit(X_polar_25)
purity_set2 = get_purity(df_25['sym_id'], km_polar.labels_)

# set 3: combined cartesian and polar coordinates (standardscaler applied)
X_combined_25 = df_25[['rx_I', 'rx_Q', 'r', 'theta']].values
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined_25)
km_combined = KMeans(n_clusters=16, init='k-means++', n_init=10, random_state=42).fit(X_combined_scaled)
purity_set3 = get_purity(df_25['sym_id'], km_combined.labels_)

# print the purity results to the console so we can document them in the report
print(f"Purity Feature Set 1 (Cartesian)  : {purity_set1:.2f}%")
print(f"Purity Feature Set 2 (Polar)      : {purity_set2:.2f}%")
print(f"Purity Feature Set 3 (Combined)   : {purity_set3:.2f}%")

# part (c): adaptive vs fixed demodulation strategies

adaptive_purity = []
fixed_purity = []

for snr in snr_levels:
    df_snr = df2[df2['snr_db'] == snr]
    X_snr = df_snr[['rx_I', 'rx_Q']].values
    
    # strategy 1: fit a fresh k-means model directly on that specific snr's cartesian data
    km_adapt = KMeans(n_clusters=16, init='k-means++', n_init=10, random_state=42).fit(X_snr)
    adaptive_purity.append(get_purity(df_snr['sym_id'], km_adapt.labels_))
    
    # strategy 2: use the fixed centroids learned strictly from the 25 db dataset
    fixed_labels = km16.predict(X_snr)
    fixed_purity.append(get_purity(df_snr['sym_id'], fixed_labels))

# plot the cluster purity of both demodulation strategies at each snr
plt.figure(figsize=(10, 5))
plt.plot(snr_levels, adaptive_purity, marker='o', label='Adaptive K-Means (Fresh Fit)')
plt.plot(snr_levels, fixed_purity, marker='^', label='Fixed Template (25 dB Centroids)')
plt.title('Part C: K-Means Cluster Purity vs SNR')
plt.xlabel('SNR (dB)')
plt.ylabel('Cluster Purity (%)')
plt.legend()
plt.grid(True)
plt.show()

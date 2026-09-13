import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# ensure generate_dataset.py has been run and los_nlos_dataset.csv exists in the directory
df = pd.read_csv('los_nlos_dataset.csv')

# ==========================================
# part (a): feature extraction
# ==========================================
# extract raw real and imaginary tap arrays across all 6 taps
h_real = df[[f'h_real_{i}' for i in range(6)]].values
h_imag = df[[f'h_imag_{i}' for i in range(6)]].values
taus = df[[f'tau_{i}' for i in range(6)]].values

# calculate instantaneous amplitude and power for each tap
amplitudes = np.sqrt(h_real**2 + h_imag**2)
powers = amplitudes**2

# 1. kurtosis and 2. skewness across the tap amplitudes
df['kurtosis'] = kurtosis(amplitudes, axis=1, fisher=False)
df['skewness'] = skew(amplitudes, axis=1)

# 3. rising time: relative delay of the tap containing the maximum power
max_power_idx = np.argmax(powers, axis=1)
tau_max_power = taus[np.arange(len(taus)), max_power_idx]
df['rising_time'] = tau_max_power - np.min(taus, axis=1)

# 4. rms delay spread: power-weighted standard deviation of tap delays
mean_delay = np.sum(taus * powers, axis=1) / np.sum(powers, axis=1)
delay_variance = np.sum(((taus - mean_delay[:, None])**2) * powers, axis=1) / np.sum(powers, axis=1)
df['rms_delay_spread'] = np.sqrt(delay_variance)

# 5. rician k-factor: ratio of dominant peak power to scattered variance
df['rician_k'] = (np.max(amplitudes, axis=1)**2) / (2 * np.var(amplitudes, axis=1))

# ==========================================
# part (b): adaptive svms (train = test snr)
# ==========================================
features = ['kurtosis', 'skewness', 'rising_time', 'rms_delay_spread', 'rician_k']
snr_levels = sorted(df['snr_db'].unique())

# dictionary to store accuracy scores for all 6 svm configurations
results_b = {f'SVM-{i+1}': [] for i in range(6)}

for snr in snr_levels:
    # isolate data for the current snr loop
    df_snr = df[df['snr_db'] == snr]
    y = df_snr['label'].values
    
    for i in range(6):
        # svms 1-5 train on single features; svm-6 trains on all 5 combined
        X = df_snr[[features[i]]].values if i < 5 else df_snr[features].values
        
        # 80/20 train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # standardize features (fit on train, transform on test to prevent data leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # train rbf svm with c=1
        clf = SVC(kernel='rbf', C=1)
        clf.fit(X_train_scaled, y_train)
        
        # record accuracy percentage
        results_b[f'SVM-{i+1}'].append(clf.score(X_test_scaled, y_test) * 100)

# plot part (b) results
plt.figure(figsize=(10, 5))
for key, acc in results_b.items():
    plt.plot(snr_levels, acc, marker='o', label=key)
plt.title('Part B: SVM Classification Accuracy vs SNR (Adaptive Training)')
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# part (c): fixed 25 db model evaluation
# ==========================================
# isolate data exclusively at 25 db for fixed training
df_25 = df[df['snr_db'] == 25]
X_train_25, _, y_train_25, _ = train_test_split(
    df_25[features].values, df_25['label'].values, test_size=0.2, random_state=42
)

# fit scaler and model strictly on 25 db training subset
scaler_fixed = StandardScaler()
X_train_25_scaled = scaler_fixed.fit_transform(X_train_25)

clf_fixed = SVC(kernel='rbf', C=1)
clf_fixed.fit(X_train_25_scaled, y_train_25)

# evaluate the fixed 25 db model across all snr subsets
fixed_acc = []
for snr in snr_levels:
    # transform features using the fixed scaler
    df_snr = df[df['snr_db'] == snr]
    X_test_all = scaler_fixed.transform(df_snr[features].values)
    
    # record evaluation accuracy
    fixed_acc.append(clf_fixed.score(X_test_all, df_snr['label'].values) * 100)

# plot part (c) results
plt.figure(figsize=(10, 5))
plt.plot(snr_levels, results_b['SVM-6'], marker='s', label='Train = Test SNR (Adaptive SVM-6)')
plt.plot(snr_levels, fixed_acc, marker='^', label='Train at 25 dB (Fixed Model)')
plt.title('Part C: Adaptive vs Fixed Training Strategies')
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()
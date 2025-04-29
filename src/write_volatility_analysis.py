import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import pickle

# Configuration
db_path = './your.db'
table_name = 'your_table'
sequence_length = 40
scaling_param_path = 'scaling_params_write.pkl'

# Load only startTime and writeRateTotal
conn = sqlite3.connect(db_path)
query = f"SELECT startTime, writeRateTotal FROM {table_name}"
df = pd.read_sql(query, conn)
conn.close()

# Convert startTime to datetime and sort
df['startTime'] = pd.to_datetime(df['startTime'])
df = df.sort_values(by='startTime').reset_index(drop=True)

# Filter out non-positive values
df = df[(df['writeRateTotal'] > 0)]

# Load scaling parameters
with open(scaling_param_path, 'rb') as f:
    params = pickle.load(f)

# Apply log and min-max scaling
df['writeRateTotal'] = np.log(df['writeRateTotal'] + 0.01)
df['writeRateTotal'] = (df['writeRateTotal'] - params['min']) / (params['max'] - params['min'])

# Create sequences
y_test_seq = df['writeRateTotal'].values
X_start_seq = df['startTime'].values[sequence_length - 1:]
y_test_seq = y_test_seq[sequence_length - 1:]

# Compute rolling standard deviation
rolling_std = np.array([np.std(y_test_seq[i:i + sequence_length]) for i in range(len(y_test_seq) - sequence_length + 1)])
rolling_std = np.append(rolling_std, [np.nan] * (sequence_length - 1)) 

# Compute z-scores
mean_rolling = np.nanmean(rolling_std)
std_rolling = np.nanstd(rolling_std)
z_scores = (rolling_std - mean_rolling) / std_rolling

# Identify high-volatility indices (Z > 2)
threshold_z = 2
threshold = mean_rolling + threshold_z * std_rolling
high_volatility_indices = np.where(z_scores > threshold_z)[0]

# Plot
plt.figure(figsize=(10, 3))
plt.plot(range(len(y_test_seq)), rolling_std, label='Rolling Std Dev')
plt.scatter(high_volatility_indices, [threshold] * len(high_volatility_indices),
            color='#A52A2A', s=10, label='High Volatility Start (Z > 2)', zorder=3)
plt.axhline(y=threshold, color='#A52A2A', linestyle='--', label='Threshold (Z = 2)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, fontsize=14, frameon=False)

tick_indices = np.linspace(0, len(X_start_seq) - 1, num=8, dtype=int)
plt.xticks(ticks=tick_indices, labels=[pd.to_datetime(X_start_seq[i]).strftime('%m-%d') for i in tick_indices])
plt.xlabel('Start Time (2018-MM-DD)')
plt.ylabel('Rolling Std Dev')
plt.yticks(np.linspace(0, 0.3, 4))
plt.tick_params(axis='both', which='both', labelsize=14)
plt.rc('axes', labelsize=18)
plt.gca().grid(True, axis='y', alpha=0.6)
plt.gca().grid(True, axis='x', alpha=0.6)
plt.tight_layout()
plt.savefig('./write_volatility.pdf')

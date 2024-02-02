import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data_file = 'tally-bench-results/priority-aware-perf.csv'
df = pd.read_csv(data_file)

df["99th_latency_increase"] = df["high_priority_tally_99th_latency"] / df["high_priority_orig_99th_latency"]

f, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

# Plot each category
categories = df['preemption_latency_limit'].unique()
categories = [float(category) for category in categories]
categories.sort()

for i, category in enumerate(categories):
    subset = df[df['preemption_latency_limit'] == category]
    sns.histplot(subset['99th_latency_increase'], ax=axes[i], kde=True)
    axes[i].set_xlim(0.8, 3)
    axes[i].set_title(f'Max allowed preemption latency (ms): {category}')

plt.tight_layout()

plt.savefig("plot.png")
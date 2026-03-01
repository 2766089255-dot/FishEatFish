import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('training_log.csv')

# Calculate moving averages (adjust window size as needed)
window = 100
df['reward_smooth'] = df['reward'].rolling(window, min_periods=1).mean()
df['steps_smooth'] = df['steps'].rolling(window, min_periods=1).mean()
df['win_rate'] = df['win'].rolling(window, min_periods=1).mean()

# Create figure with three subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Total reward curve
axes[0].plot(df['episode'], df['reward'], alpha=0.3, label='Raw Reward')
axes[0].plot(df['episode'], df['reward_smooth'], 'r-', label=f'{window}-Episode Moving Avg')
axes[0].set_ylabel('Total Reward')
axes[0].legend()
axes[0].grid(True)

# Steps per episode curve
axes[1].plot(df['episode'], df['steps'], alpha=0.3, label='Raw Steps')
axes[1].plot(df['episode'], df['steps_smooth'], 'b-', label=f'{window}-Episode Moving Avg')
axes[1].set_ylabel('Steps per Episode')
axes[1].legend()
axes[1].grid(True)

# Win rate curve
axes[2].plot(df['episode'], df['win_rate'], 'g-', label='Win Rate (moving avg)')
axes[2].set_xlabel('Episode')
axes[2].set_ylabel('Win Rate')
axes[2].set_ylim(0, 1)
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
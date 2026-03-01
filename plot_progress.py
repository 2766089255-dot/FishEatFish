import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
df = pd.read_csv('training_log.csv')

# 计算滑动平均（窗口大小可调）
window = 100
df['reward_smooth'] = df['reward'].rolling(window, min_periods=1).mean()
df['steps_smooth'] = df['steps'].rolling(window, min_periods=1).mean()
df['win_rate'] = df['win'].rolling(window, min_periods=1).mean()

# 创建图表
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# 总奖励曲线
axes[0].plot(df['episode'], df['reward'], alpha=0.3, label='Raw Reward')
axes[0].plot(df['episode'], df['reward_smooth'], 'r-', label=f'{window}-Episode Moving Avg')
axes[0].set_ylabel('Total Reward')
axes[0].legend()
axes[0].grid(True)

# 步数曲线
axes[1].plot(df['episode'], df['steps'], alpha=0.3, label='Raw Steps')
axes[1].plot(df['episode'], df['steps_smooth'], 'b-', label=f'{window}-Episode Moving Avg')
axes[1].set_ylabel('Steps per Episode')
axes[1].legend()
axes[1].grid(True)

# 胜率曲线
axes[2].plot(df['episode'], df['win_rate'], 'g-', label='Win Rate (moving avg)')
axes[2].set_xlabel('Episode')
axes[2].set_ylabel('Win Rate')
axes[2].set_ylim(0, 1)
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
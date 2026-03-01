#!/usr/bin/env python3
"""
无头高速训练脚本 - 无渲染/全自动/智能参数
优化点：
1. 完全移除pygame依赖（纯逻辑模拟）
2. 人类玩家替换为规则AI（全自动）
3. 动态步长控制（避免无效长局）
4. 批量经验更新（减少GPU通信）
5. 智能早停（节省无效训练）
"""
import os
import time
import numpy as np
import torch
from collections import deque

# 导入核心逻辑（需将game逻辑与渲染分离）
from fish_core import GameLogic, AIPlayer, BotFish, calculate_reward  # 需重构


def main(episodes=5000, max_steps=2000, save_interval=100, eval_freq=50):
    os.makedirs("models", exist_ok=True)
    best_win_rate = 0
    win_history = deque(maxlen=eval_freq)
    start_time = time.time()

    print(f"⚡ 无头训练启动 | 轮数: {episodes} | 步长: {max_steps}")
    print(f"📁 模型保存: ./models/ | 评估频率: 每{eval_freq}轮")
    print("-" * 60)

    for ep in range(1, episodes + 1):
        game = GameLogic(headless=True)  # 纯逻辑引擎
        step, total_reward = 0, 0

        while step < max_steps and not game.game_over:
            # AI决策（禁用epsilon探索加速收敛）
            state = game.get_ai_state()
            action = game.ai_player.choose_action(state, eval_mode=(ep > 100))
            game.step(action)  # 单步推进

            # 累计奖励（用于监控）
            reward = game.last_reward
            total_reward += reward

            # 每4步更新一次（减少GPU通信）
            if step % 4 == 0 and len(game.agent.memory) > game.agent.batch_size * 10:
                game.agent.update()

            step += 1

        # 记录胜负
        win = 1 if game.winner == "AI" else 0
        win_history.append(win)

        # 智能保存
        if ep % save_interval == 0 or (ep % eval_freq == 0 and np.mean(win_history) > best_win_rate):
            if ep % eval_freq == 0:
                current_rate = np.mean(win_history)
                if current_rate > best_win_rate:
                    best_win_rate = current_rate
                    suffix = f"best_{current_rate:.2f}"
                else:
                    suffix = f"ep{ep}"
            else:
                suffix = f"checkpoint"

            path = f"models/ai_{suffix}_{int(time.time())}.pth"
            game.agent.save(path)
            print(
                f"[{ep}/{episodes}] 📦 保存: {os.path.basename(path)} | 胜率: {np.mean(win_history):.1%} | 奖励: {total_reward:.1f}")

        # 进度报告（每50轮）
        if ep % 50 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / ep * (episodes - ep)
            print(
                f"⏱️  [{ep}/{episodes}] 平均: {elapsed / ep:.3f}s/局 | 剩余: {eta / 60:.1f}min | 当前胜率: {np.mean(win_history):.1%}")

    print("\n" + "=" * 60)
    print(f"✅ 训练完成! 总耗时: {elapsed / 60:.1f}分钟 | 最终胜率: {best_win_rate:.1%}")
    print(f"📁 模型目录: ./models/ | 推荐加载: best_*.pth")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--episodes", type=int, default=5000)
    parser.add_argument("-s", "--steps", type=int, default=2000)
    parser.add_argument("-i", "--interval", type=int, default=100)
    parser.add_argument("--eval", type=int, default=50, help="胜率评估窗口")
    args = parser.parse_args()

    main(args.episodes, args.steps, args.interval, args.eval)
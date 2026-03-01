#!/usr/bin/env python3
"""精简训练脚本 - 100%兼容现有game.py"""
import os
import time
import argparse
from game import Game


def main(episodes=10, save_interval=5):
    os.makedirs("models", exist_ok=True)
    print(f"🚀 启动训练 | 轮数: {episodes} | 保存间隔: {save_interval}")
    print("💡 每局结束后请手动关闭游戏窗口继续下一轮")
    print("-" * 50)

    for ep in range(1, episodes + 1):
        print(f"\n▶️  轮次 [{ep}/{episodes}] - 请操作人类玩家（方向键）")
        game = Game()
        game.run()  # 游戏窗口会弹出，玩完后关闭窗口

        # 保存模型（每N轮）
        if ep % save_interval == 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = f"models/ai_ep{ep}_{timestamp}.pth"
            game.save_model(path)  # 调用我们动态添加的方法

        # 进度提示
        print(f"⏱️  已完成 {ep}/{episodes} 轮 | 按Ctrl+C可随时终止训练")

    print("\n" + "=" * 50)
    print(f"🎉 训练完成！模型保存在: {os.path.abspath('models')}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--episodes", type=int, default=10, help="训练轮数")
    parser.add_argument("-i", "--interval", type=int, default=5, help="保存间隔")
    args = parser.parse_args()
    main(args.episodes, args.interval)
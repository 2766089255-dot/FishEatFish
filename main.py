# main.py
# Entry point for training and human vs AI gameplay

import sys
from game import Game
from ai_agent import DQNAgent

if __name__ == '__main__':
    training_mode = False
    render = False
    model_path = None
    target_episodes = None
    fast_mode = False
    resume_path = None
    start_episode = 0

    # Parse command line arguments
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--train':
            training_mode = True
        elif arg == '--render':
            render = True
        elif arg == '--fast':
            fast_mode = True
        elif arg == '--load' and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            i += 1
        elif arg == '--resume' and i + 1 < len(sys.argv):
            resume_path = sys.argv[i + 1]
            i += 1
        elif arg == '--start-episode' and i + 1 < len(sys.argv):
            start_episode = int(sys.argv[i + 1])
            i += 1
        elif arg == '--episodes' and i + 1 < len(sys.argv):
            target_episodes = int(sys.argv[i + 1])
            i += 1
        i += 1

    # Create game instance
    game = Game(training_mode=training_mode, render=render,
                target_episodes=target_episodes, fast_mode=fast_mode)

    # Resume training: load model and set starting episode
    if resume_path and training_mode:
        state_dim = 4 + 10 * 5
        agent = DQNAgent(state_dim=state_dim, action_dim=8)
        agent.load(resume_path)
        game.set_ai_agent(agent)
        game.episode = start_episode
        print(f"Resuming training: loaded {resume_path}, starting from episode {start_episode}")

    # Load model for evaluation (human vs AI)
    elif model_path and not training_mode:
        state_dim = 4 + 10 * 5
        agent = DQNAgent(state_dim=state_dim, action_dim=8)
        agent.load(model_path)
        game.set_ai_agent(agent)
        print(f"Model {model_path} loaded successfully. AI will use this model (evaluation mode).")

    game.run()
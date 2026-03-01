# game.py
# Main game class managing environment, players, and training loop

import pygame
import random
import csv
import os
from config import *
from player import AIPlayer, HumanPlayer
from bot import BotFish
from utils import truncated_normal_prob, sample_level
from ai_agent import DQNAgent

class Game:
    def __init__(self, training_mode=False, render=False, target_episodes=None, fast_mode=False):
        """
        Initialize the game.
        Args:
            training_mode: if True, only AI player exists and training is active
            render: if True, draw the game screen
            target_episodes: if training, stop after this many episodes
            fast_mode: if True, no waiting in training loop (max speed)
        """
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Big Fish Eat Small Fish - Human vs AI")
        self.clock = pygame.time.Clock()
        self.running = True
        self.game_over = False
        self.step_count = 0
        self.winner = None
        self.training_mode = training_mode
        self.render = render
        self.target_episodes = target_episodes
        self.fast_mode = fast_mode
        self.episode = 0                     # number of completed episodes
        self.rewards_history = []             # store total reward per episode
        self.level_probs = None                # probability distribution for bot levels

        if training_mode:
            # Training mode: only AI player with agent
            state_dim = 4 + 10 * 5              # 4 self + 10 bots * 5
            self.agent = DQNAgent(state_dim=state_dim, action_dim=8)
            self.ai_player = AIPlayer(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, agent=self.agent)
            self.human_player = None
        else:
            # Human vs AI mode: both players, AI may have agent loaded later
            self.ai_player = AIPlayer(2 * SCREEN_WIDTH // 3, SCREEN_HEIGHT // 2)
            self.human_player = HumanPlayer(SCREEN_WIDTH // 3, SCREEN_HEIGHT // 2)
            self.agent = None

        self.bots = []
        self.update_level_distribution()

        # CSV logging for training data
        self.csv_file = 'training_log.csv'
        file_exists = os.path.isfile(self.csv_file)
        self.csv_handle = open(self.csv_file, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_handle)
        if not file_exists:
            self.csv_writer.writerow(['episode', 'reward', 'steps', 'final_level', 'win'])

    def set_ai_agent(self, agent):
        """Assign a DQN agent to the AI player."""
        self.ai_player.agent = agent

    def update_level_distribution(self):
        """Update bot level distribution centered at (AI level + 1)."""
        # In training mode, center is AI level+1; in dual mode, use max of both players
        if self.training_mode:
            base_level = self.ai_player.level
        else:
            if self.human_player:
                base_level = max(self.ai_player.level, self.human_player.level)
            else:
                base_level = self.ai_player.level
        center = min(base_level + 1, MAX_LEVEL - 1)   # cap at max bot level (MAX_LEVEL-1)
        self.level_probs = truncated_normal_prob(center)

    def get_state_for_ai(self):
        """
        Construct the state vector for the AI player.
        Contains self info and features of nearest 10 bots.
        """
        ai = self.ai_player
        state = [
            ai.level / MAX_LEVEL,
            ai.hp / PLAYER_INIT_HP,
            ai.x / SCREEN_WIDTH,
            ai.y / SCREEN_HEIGHT
        ]

        N = 10
        # Compute distances to all bots
        bots_with_dist = [(bot, ((bot.x - ai.x) ** 2 + (bot.y - ai.y) ** 2)) for bot in self.bots]
        bots_with_dist.sort(key=lambda x: x[1])
        nearest_bots = bots_with_dist[:N]

        for bot, _ in nearest_bots:
            state.extend([
                (bot.x - ai.x) / SCREEN_WIDTH,
                (bot.y - ai.y) / SCREEN_HEIGHT,
                bot.level / MAX_LEVEL,
                bot.dx / BOT_SPEED,
                bot.dy / BOT_SPEED
            ])
        # Pad with zeros if fewer than N bots
        for _ in range(N - len(nearest_bots)):
            state.extend([0, 0, 0, 0, 0])

        return state

    def spawn_bot(self):
        """Generate a new bot fish if conditions allow."""
        if len(self.bots) >= MAX_BOT_COUNT:
            return
        if random.random() > BOT_GEN_PROB:
            return
        level = sample_level(self.level_probs)
        bot = BotFish(level)
        self.bots.append(bot)

    def check_collisions_for_player(self, player):
        """
        Handle collisions between a given player and all bot fish.
        Returns True if the player ate any fish.
        """
        ate_fish = False
        for bot in self.bots[:]:
            dist = ((player.x - bot.x) ** 2 + (player.y - bot.y) ** 2) ** 0.5
            if dist < player.radius + bot.radius:
                if player.level >= bot.level:
                    # Player can eat this bot
                    if player.level > bot.level:
                        level_diff = player.level - bot.level
                        exp_gain = 1.0 / level_diff   # lower level gives less XP
                    else:
                        exp_gain = 1.0                # same level gives 1 XP
                    player.exp_progress += exp_gain
                    self.bots.remove(bot)
                    ate_fish = True
                    prefix = "AI" if player is self.ai_player else "Human"
                    print(f"{prefix} ate level {bot.level} fish! XP +{exp_gain:.2f}")

                    # Level up if enough XP accumulated
                    while player.exp_progress >= 1.0 and player.level < MAX_LEVEL:
                        player.upgrade()
                        self.update_level_distribution()
                        print(f"{prefix} leveled up to {player.level}!")
                        if player.level >= MAX_LEVEL:
                            self.game_over = True
                            if player is self.ai_player:
                                self.winner = 'ai'
                            elif self.human_player and player is self.human_player:
                                self.winner = 'human'
                            else:
                                self.winner = None
                            break
                elif player.level < bot.level:
                    # Player is hit by larger fish
                    if player.take_damage():
                        self.game_over = True
                        if player is self.ai_player:
                            self.winner = 'human'
                        elif self.human_player and player is self.human_player:
                            self.winner = 'ai'
                        else:
                            self.winner = None
                    # bot does not disappear
        return ate_fish

    def update(self):
        """Main update logic: handle input, AI decisions, bot updates, collisions."""
        if self.game_over:
            return
        self.step_count += 1

        # For training mode, record old stats for reward calculation
        if self.training_mode:
            old_level = self.ai_player.level
            old_hp = self.ai_player.hp
            old_bot_count = len(self.bots)
        else:
            old_level = old_hp = old_bot_count = 0   # dummy values

        # Handle human player input (if exists)
        if not self.training_mode and self.human_player:
            keys = pygame.key.get_pressed()
            self.human_player.handle_input(keys)
            self.human_player.update_invincible()

        # AI decision
        if self.training_mode:
            state = self.get_state_for_ai()
            action = self.ai_player.choose_action(state, eval_mode=False)
        else:
            if self.ai_player.agent is not None:
                state = self.get_state_for_ai()
                action = self.ai_player.choose_action(state, eval_mode=True)
                # Optional debug: print(f"[EVAL] action={action}")
            else:
                import random
                action = random.choice([0, 1, 2, 3, 4, 5, 6, 7])
        self.ai_player.apply_action(action)
        self.ai_player.update_invincible()

        # Spawn new bots
        self.spawn_bot()

        # Update all bots
        for bot in self.bots[:]:
            bot.update()
            if bot.is_offscreen():
                self.bots.remove(bot)

        # Collision detection for all players
        if self.training_mode:
            self.check_collisions_for_player(self.ai_player)
        else:
            if self.human_player:
                self.check_collisions_for_player(self.human_player)
            self.check_collisions_for_player(self.ai_player)

        # Training mode: compute reward and store experience
        if self.training_mode:
            # Step penalty
            reward = -0.05

            # Eating reward
            if len(self.bots) < old_bot_count:
                reward += 30.0

            # Level-up reward
            level_gain = self.ai_player.level - old_level
            if level_gain > 0:
                reward += 50.0 * level_gain

            # Damage penalty
            hp_loss = old_hp - self.ai_player.hp
            if hp_loss > 0:
                reward -= 0.2 * hp_loss

            # Boundary penalty (discourage staying near edges)
            margin = 50
            dist_left = self.ai_player.x - self.ai_player.radius
            dist_right = SCREEN_WIDTH - self.ai_player.x - self.ai_player.radius
            dist_top = self.ai_player.y - self.ai_player.radius
            dist_bottom = SCREEN_HEIGHT - self.ai_player.y - self.ai_player.radius
            min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
            if min_dist < margin:
                edge_penalty = -0.5 * (margin - min_dist) / margin
                reward += edge_penalty

            # Terminal rewards
            if self.game_over:
                if self.ai_player.level >= MAX_LEVEL:
                    reward += 100
                    print(f"Episode {self.episode+1} ended: VICTORY! steps={self.step_count}, total reward={reward:.2f}, final level={self.ai_player.level}")
                else:
                    reward -= 10
                    print(f"Episode {self.episode+1} ended: DEFEAT! steps={self.step_count}, total reward={reward:.2f}, final level={self.ai_player.level}")
                self.rewards_history.append(reward)
                next_state = [0.0] * self.agent.state_dim
            else:
                next_state = self.get_state_for_ai()

            done = self.game_over
            self.agent.store_transition(state, action, reward, next_state, done)
            self.agent.update()

            # Periodic status print
            if self.step_count % 100 == 0:
                print(f"Step {self.step_count}: AI level {self.ai_player.level}, bots around {len(self.bots)}")

    def reset_game(self):
        """Reset the game for a new episode (both players and bots)."""
        print("Resetting game, starting new episode.")
        if self.training_mode:
            self.ai_player = AIPlayer(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, agent=self.agent)
            self.human_player = None
        else:
            self.ai_player = AIPlayer(2 * SCREEN_WIDTH // 3, SCREEN_HEIGHT // 2, agent=self.ai_player.agent)
            self.human_player = HumanPlayer(SCREEN_WIDTH // 3, SCREEN_HEIGHT // 2)
        self.bots.clear()
        self.game_over = False
        self.step_count = 0
        self.winner = None
        self.update_level_distribution()

    def draw(self):
        """Render the game screen."""
        if not self.render:
            return

        self.draw_gradient_background()

        if self.human_player:
            self.human_player.draw(self.screen)
        self.ai_player.draw(self.screen)

        for bot in self.bots:
            bot.draw(self.screen)

        font = pygame.font.SysFont(None, 24)

        # Display human and AI levels with HP below to avoid overlap
        if self.human_player:
            # Human level
            human_level_text = font.render(f"Human Level: {self.human_player.level}", True, GREEN)
            self.screen.blit(human_level_text, (10, 10))
            # Human HP (placed below level)
            human_hp_text = font.render(f"HP: {self.human_player.hp}", True, RED)
            self.screen.blit(human_hp_text, (10, 35))
            # AI level (further down)
            ai_level_text = font.render(f"AI Level: {self.ai_player.level}", True, BLUE)
            self.screen.blit(ai_level_text, (10, 60))
        else:
            # Only AI exists (training mode)
            ai_level_text = font.render(f"AI Level: {self.ai_player.level}", True, BLUE)
            self.screen.blit(ai_level_text, (10, 10))

        # Game over overlay
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            if self.winner == 'ai':
                msg = "AI WINS!"
                color = BLUE
            elif self.winner == 'human':
                msg = "HUMAN WINS!"
                color = GREEN
            else:
                msg = "GAME OVER"
                color = RED
            text = font.render(msg, True, color)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(text, text_rect)

        pygame.display.flip()

    def draw_gradient_background(self):
        """Draw vertical gradient from top to bottom."""
        for y in range(SCREEN_HEIGHT):
            ratio = y / SCREEN_HEIGHT
            r = int(SKY_BLUE_TOP[0] * (1 - ratio) + SKY_BLUE_BOTTOM[0] * ratio)
            g = int(SKY_BLUE_TOP[1] * (1 - ratio) + SKY_BLUE_BOTTOM[1] * ratio)
            b = int(SKY_BLUE_TOP[2] * (1 - ratio) + SKY_BLUE_BOTTOM[2] * ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))

    def start_screen(self):
        """Display start screen with English rules, lasts 3 seconds or skip by key press."""
        start_ticks = pygame.time.get_ticks()
        waiting = True
        font_title = pygame.font.SysFont(None, 72)
        font_text = pygame.font.SysFont(None, 32)
        font_small = pygame.font.SysFont(None, 24)

        title_text = "Big Fish Eat Small Fish"
        rule_lines = [
            "Rules:",
            "- Eat fish of lower or equal level to gain XP.",
            "- Level up when XP reaches 1.0.",
            "- Colliding with a larger fish costs 1 HP.",
            "- First player to reach level 15 wins.",
            "",
            "Controls:",
            "- Player (Green): Arrow keys",
            "- AI (Blue): Autonomous"
        ]
        continue_text = "Auto start in 3 seconds or press any key"

        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                if event.type == pygame.KEYDOWN:
                    waiting = False

            elapsed = pygame.time.get_ticks() - start_ticks
            if elapsed > START_SCREEN_DURATION:
                waiting = False

            self.screen.fill(BG_COLOR)

            title_surf = font_title.render(title_text, True, TITLE_COLOR)
            title_rect = title_surf.get_rect(center=(SCREEN_WIDTH//2, 80))
            self.screen.blit(title_surf, title_rect)

            y_offset = 150
            for line in rule_lines:
                if line.startswith("-"):
                    color = TEXT_COLOR
                elif line == "":
                    y_offset += 10
                    continue
                else:
                    color = TITLE_COLOR
                text_surf = font_text.render(line, True, color)
                text_rect = text_surf.get_rect(center=(SCREEN_WIDTH//2, y_offset))
                self.screen.blit(text_surf, text_rect)
                y_offset += 40

            cont_surf = font_small.render(continue_text, True, TEXT_COLOR)
            cont_rect = cont_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT-50))
            self.screen.blit(cont_surf, cont_rect)

            pygame.display.flip()
            self.clock.tick(60)

    def run(self):
        """Main game loop."""
        if not self.training_mode:
            self.start_screen()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if not self.training_mode and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and self.game_over:
                        self.reset_game()

            self.update()

            if self.render:
                self.draw()
                self.clock.tick(FPS)
            else:
                if self.fast_mode:
                    pass          # no waiting, maximum speed
                else:
                    pygame.time.wait(1)

            # Training mode: auto reset when game over
            if self.training_mode and self.game_over:
                self.episode += 1
                # Save model every 1000 episodes
                if self.episode % 1000 == 0:
                    self.agent.save(f"model_ep{self.episode}.pth")
                    print(f"Model saved to model_ep{self.episode}.pth")
                # Write episode data to CSV
                win_flag = 1 if self.ai_player.level >= MAX_LEVEL else 0
                self.csv_writer.writerow([self.episode, self.rewards_history[-1] if self.rewards_history else 0,
                                          self.step_count, self.ai_player.level, win_flag])
                self.csv_handle.flush()
                self.reset_game()

                if self.target_episodes is not None and self.episode >= self.target_episodes:
                    print(f"Reached target episodes {self.target_episodes}, training finished.")
                    self.running = False

        pygame.quit()
        self.csv_handle.close()
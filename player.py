# player.py
# Player fish classes (Human, AI)

import pygame
from fish import Fish
from config import *

class BasePlayerFish(Fish):
    """Base class for all player-controlled fish (human and AI)."""
    def __init__(self, x, y, level, speed, radius_base, radius_per_level, hp):
        super().__init__(x, y, level, speed, radius_base, radius_per_level)
        self.hp = hp
        self.invincible = 0                     # frames remaining invincible
        self.radius_base = radius_base
        self.radius_per_level = radius_per_level
        self.exp_progress = 0.0                  # current experience towards next level

    def update_invincible(self):
        """Decrease invincibility counter each frame."""
        if self.invincible > 0:
            self.invincible -= 1

    def upgrade(self):
        """Increase level by 1, reset HP to 3, and update size."""
        if self.level >= MAX_LEVEL:
            return
        self.level += 1
        self.hp = 3
        self.radius = self.radius_base + self.level * self.radius_per_level
        max_radius = min(SCREEN_WIDTH, SCREEN_HEIGHT) // 3
        if self.radius > max_radius:
            self.radius = max_radius
        self.exp_progress -= 1.0

    def take_damage(self):
        """
        Lose 1 HP if not invincible.
        Returns True if HP reaches zero (death).
        """
        if self.invincible > 0:
            return False
        self.hp -= 1
        self.invincible = INVINCIBLE_DURATION
        return self.hp <= 0

    def draw(self, screen, color):
        """
        Draw the fish with optional flashing when invincible.
        """
        draw_color = color
        if self.invincible > 0 and (self.invincible // 5) % 2 == 0:
            # Flash by lightening color
            draw_color = tuple(min(255, c + 100) for c in color)
        super().draw(screen, draw_color)
        # Draw level number
        font = pygame.font.SysFont(None, 20)
        text = font.render(str(self.level), True, BLACK)
        screen.blit(text, (self.x - 10, self.y - self.radius - 20))


class HumanPlayer(BasePlayerFish):
    """Human player controlled by arrow keys."""
    def __init__(self, x, y):
        super().__init__(x, y, PLAYER_INIT_LEVEL, PLAYER_SPEED,
                         PLAYER_RADIUS_BASE, PLAYER_RADIUS_PER_LEVEL, PLAYER_INIT_HP)

    def handle_input(self, keys):
        """Move based on arrow key states."""
        dx = dy = 0
        if keys[pygame.K_LEFT]:
            dx = -self.speed
        if keys[pygame.K_RIGHT]:
            dx = self.speed
        if keys[pygame.K_UP]:
            dy = -self.speed
        if keys[pygame.K_DOWN]:
            dy = self.speed
        self.move(dx, dy)

    def draw(self, screen):
        """Draw human fish in green and show HP."""
        super().draw(screen, GREEN)
        font = pygame.font.SysFont(None, 24)
        hp_text = font.render(f"HP: {self.hp}", True, RED)
        screen.blit(hp_text, (10, 10))


class AIPlayer(BasePlayerFish):
    """AI player controlled by a DQN agent."""
    def __init__(self, x, y, agent=None):
        super().__init__(x, y, PLAYER_INIT_LEVEL, PLAYER_SPEED,
                         PLAYER_RADIUS_BASE, PLAYER_RADIUS_PER_LEVEL, PLAYER_INIT_HP)
        self.agent = agent      # reference to the DQN agent

    def choose_action(self, state, eval_mode=False):
        """
        Select an action based on current state.
        If eval_mode is True, use greedy policy; otherwise use epsilon-greedy.
        """
        if self.agent:
            return self.agent.select_action(state, eval_mode=eval_mode)
        else:
            import random
            return random.choice([0, 1, 2, 3, 4, 5, 6, 7])

    def apply_action(self, action):
        """Execute the given action (0-7 corresponding to 8 directions)."""
        dx = dy = 0
        if action == 0:   # up
            dy = -self.speed
        elif action == 1: # down
            dy = self.speed
        elif action == 2: # left
            dx = -self.speed
        elif action == 3: # right
            dx = self.speed
        elif action == 4: # up-left
            dx = -self.speed
            dy = -self.speed
        elif action == 5: # up-right
            dx = self.speed
            dy = -self.speed
        elif action == 6: # down-left
            dx = -self.speed
            dy = self.speed
        elif action == 7: # down-right
            dx = self.speed
            dy = self.speed
        old_x, old_y = self.x, self.y
        self.move(dx, dy)
        # Optional debug print (can be disabled)
        # print(f"Action {action}: ({old_x:.1f},{old_y:.1f}) -> ({self.x:.1f},{self.y:.1f})")

    def draw(self, screen):
        """Draw AI fish in blue."""
        super().draw(screen, BLUE)
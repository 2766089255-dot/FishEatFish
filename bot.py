# bot.py
# Bot fish class (controlled by simple linear movement)

import random
from fish import Fish
from config import *

class BotFish(Fish):
    """Bot fish that moves linearly from screen edges."""
    def __init__(self, level):
        """
        Create a bot fish at a random screen edge with a given level.
        The fish moves straight across the screen.
        """
        # Calculate temporary radius for initial positioning
        temp_radius = BOT_RADIUS_BASE + level * BOT_RADIUS_PER_LEVEL
        max_radius = min(SCREEN_WIDTH, SCREEN_HEIGHT) // 3
        if temp_radius > max_radius:
            temp_radius = max_radius

        side = random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            x = random.randint(BOT_RADIUS_BASE, SCREEN_WIDTH - BOT_RADIUS_BASE)
            y = -temp_radius
            dx, dy = 0, BOT_SPEED
        elif side == 'bottom':
            x = random.randint(BOT_RADIUS_BASE, SCREEN_WIDTH - BOT_RADIUS_BASE)
            y = SCREEN_HEIGHT + temp_radius
            dx, dy = 0, -BOT_SPEED
        elif side == 'left':
            x = -temp_radius
            y = random.randint(BOT_RADIUS_BASE, SCREEN_HEIGHT - BOT_RADIUS_BASE)
            dx, dy = BOT_SPEED, 0
        else:  # right
            x = SCREEN_WIDTH + temp_radius
            y = random.randint(BOT_RADIUS_BASE, SCREEN_HEIGHT - BOT_RADIUS_BASE)
            dx, dy = -BOT_SPEED, 0

        super().__init__(x, y, level, BOT_SPEED,
                         BOT_RADIUS_BASE, BOT_RADIUS_PER_LEVEL)
        self.dx = dx
        self.dy = dy

    def update(self):
        """Move the bot fish according to its direction."""
        self.x += self.dx
        self.y += self.dy

    def is_offscreen(self):
        """Check if the bot has moved completely off the screen."""
        return (self.x + self.radius < 0 or self.x - self.radius > SCREEN_WIDTH or
                self.y + self.radius < 0 or self.y - self.radius > SCREEN_HEIGHT)

    def draw(self, screen):
        """Draw the bot fish in red."""
        super().draw(screen, RED)
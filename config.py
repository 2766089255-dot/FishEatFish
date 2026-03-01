# config.py
# Game configuration and constants

import pygame

# Screen settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# Player initial attributes
PLAYER_INIT_LEVEL = 2
PLAYER_INIT_HP = 3
PLAYER_SPEED = 3
PLAYER_RADIUS_BASE = 10
PLAYER_RADIUS_PER_LEVEL = 2

# Bot fish attributes
BOT_SPEED = 2
BOT_RADIUS_BASE = 8
BOT_RADIUS_PER_LEVEL = 1.5
BOT_GEN_PROB = 0.5          # Probability of spawning a bot each frame
MAX_BOT_COUNT = 100         # Maximum number of bots on screen

# Level range
MIN_LEVEL = 1
MAX_LEVEL = 15              # Maximum level to win (can be changed)

# Level distribution parameter: centered at (AI level + 1), sigma controls concentration
LEVEL_DIST_SIGMA = 1.5

# Invincibility duration (frames)
INVINCIBLE_DURATION = 60

# Gradient background colors
SKY_BLUE_TOP = (173, 216, 230)      # Light blue at top
SKY_BLUE_BOTTOM = (70, 130, 180)    # Steel blue at bottom

# Start screen parameters
START_SCREEN_DURATION = 3000        # milliseconds (3 seconds)
TITLE_COLOR = (255, 215, 0)          # Gold
TEXT_COLOR = (255, 255, 255)         # White
BG_COLOR = (0, 0, 50)                # Dark blue background
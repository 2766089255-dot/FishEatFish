# fish.py
# Base class for all fish (player, AI, bot)

import pygame
from config import *

class Fish:
    """Base class for all fish entities."""
    def __init__(self, x, y, level, speed, radius_base, radius_per_level):
        """
        Initialize a fish.
        Args:
            x, y: position
            level: current level (1-30)
            speed: movement speed
            radius_base: base radius for level 1
            radius_per_level: radius increment per level
        """
        self.x = x
        self.y = y
        self.level = level
        self.speed = speed
        self.radius = radius_base + level * radius_per_level
        # Limit maximum radius to 1/3 of screen dimension
        max_radius = min(SCREEN_WIDTH, SCREEN_HEIGHT) // 3
        if self.radius > max_radius:
            self.radius = max_radius

    def move(self, dx, dy):
        """Move the fish by (dx, dy) and keep it within screen boundaries."""
        self.x += dx
        self.y += dy
        self.x = max(self.radius, min(SCREEN_WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(SCREEN_HEIGHT - self.radius, self.y))

    def get_rect(self):
        """Return a pygame Rect for collision detection."""
        return pygame.Rect(self.x - self.radius, self.y - self.radius,
                           self.radius * 2, self.radius * 2)

    def draw(self, screen, color):
        """
        Draw the fish as an ellipse with eyes and tail.
        Args:
            screen: pygame surface
            color: fish color
        """
        # Body (ellipse: width 1.5*radius, height radius)
        body_width = self.radius * 1.5
        body_height = self.radius
        body_rect = pygame.Rect(0, 0, body_width, body_height)
        body_rect.center = (self.x, self.y)
        pygame.draw.ellipse(screen, color, body_rect)

        # Eye (black pupil with white highlight)
        eye_radius = self.radius // 4
        eye_x = self.x + body_width * 0.25
        eye_y = self.y - body_height * 0.2
        pygame.draw.circle(screen, BLACK, (int(eye_x), int(eye_y)), eye_radius)
        pygame.draw.circle(screen, WHITE, (int(eye_x - 2), int(eye_y - 2)), eye_radius // 2)

        # Tail (triangle on the left side)
        tail_length = body_height * 0.8
        tail_points = [
            (self.x - body_width // 2, self.y - body_height // 2),
            (self.x - body_width // 2, self.y + body_height // 2),
            (self.x - body_width // 2 - tail_length // 2, self.y)
        ]
        pygame.draw.polygon(screen, color, tail_points)

        # Display level number above the fish
        font = pygame.font.SysFont(None, 20)
        text = font.render(str(self.level), True, BLACK)
        screen.blit(text, (self.x - 10, self.y - self.radius - 20))
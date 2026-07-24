import itertools
import os
from typing import TYPE_CHECKING, Optional

import pygame

from asteroid import Asteroid
from asteroidfield import AsteroidField
from constants import *
from explosion import Explosion
from player import Player
from shot import Shot
from utils import init_text, is_colliding

if TYPE_CHECKING:
    import ai.nn


class Game:

    def __init__(self, visual: bool = True):
        self.visual = visual

        if visual:
            pygame.init()
        else:
            # Use a dummy display/audio backend so training can run headlessly.
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
            pygame.init()

        self.updateables = pygame.sprite.Group()
        self.drawables = pygame.sprite.Group()
        self.asteroids = pygame.sprite.Group()
        self.shots = pygame.sprite.Group()

        Player.containers = (self.updateables, self.drawables)
        AsteroidField.containers = (self.updateables,)
        Asteroid.containers = (self.updateables, self.drawables, self.asteroids)
        Shot.containers = (self.updateables, self.drawables, self.shots)

        self.player = Player(x=SCREEN_WIDTH / 2, y=SCREEN_HEIGHT / 2)
        AsteroidField()

    def start(
        self,
        ship_ai: Optional["ai.nn.NeuralNetwork"] = None,
    ):
        """Visual game loop."""
        if not self.visual:
            raise RuntimeError("Game.start() requires visual=True")

        init_text()

        dt = 0
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        while True:
            if any(event.type == pygame.QUIT for event in pygame.event.get()):
                return

            if not self.player.alive():
                print("Game over!")
                print(f"Score: {self.player.score}")
                print(f"Accuracy: {self.player.accuracy:.1f}%")
                return

            if ship_ai:
                self.ai_move(ship_ai, dt)

            screen.fill("black")
            self._update_sprites(dt=dt, visual_effects=True)
            for drawable in self.drawables:
                drawable.draw(screen=screen)

            pygame.display.set_caption(f"FPS: {clock.get_fps():.1f}")
            pygame.display.flip()
            dt = clock.tick(60) / 1000

    def sim(
        self,
        ship_ai: "ai.nn.NeuralNetwork",
        dt: float = 0.05,
    ) -> tuple[int, float]:
        """
        Headless accelerated simulation.
        Used by genetic training.
        """

        for frame in itertools.count():

            if not self.player.alive():
                return (frame, self.player.score)

            self.ai_move(ship_ai, dt)
            self._update_sprites(dt=dt, visual_effects=False)

        raise RuntimeError("unreachable")

    def ai_move(
        self,
        ship_ai: "ai.nn.NeuralNetwork",
        dt: float,
    ) -> list["ai.nn.Action"]:
        from game_state import get_game_state

        if not self.asteroids.sprites():
            return []

        game_state = get_game_state(self)
        if not game_state:
            return []

        actions = ship_ai.predict(game_state)

        for action in actions:

            if action == 0:
                self.player.move(dt=dt)
            elif action == 1:
                self.player.rotate(dt=dt)
            elif action == 2:
                self.player.rotate(dt=-dt)
            elif action == 3:
                self.player.shoot()

        return actions

    def _update_sprites(self, dt: float, visual_effects: bool = True):
        self.updateables.update(dt=dt)

        asteroids = self.asteroids.sprites()
        for i, asteroid in enumerate(asteroids):
            if is_colliding(asteroid, self.player):
                points = asteroid.resolve_collision(obj=self.player) or 0
                if visual_effects:
                    explosion = Explosion(
                        position=asteroid.position, radius=asteroid.radius
                    )
                    self.updateables.add(explosion)
                    self.drawables.add(explosion)
                self.player.respawn(points_lost=points)
            for other in asteroids[i + 1 :]:
                if is_colliding(asteroid, other):
                    asteroid.resolve_collision(other)
            for shot in self.shots:
                if is_colliding(asteroid, shot):
                    self.player.shots_hit += 1
                    self.player.score += asteroid.split(damage=shot.damage)
                    if not asteroid.alive() and visual_effects:
                        explosion = Explosion(
                            position=asteroid.position, radius=asteroid.radius
                        )
                        self.updateables.add(explosion)
                        self.drawables.add(explosion)
                    shot.kill()

    def load_ai(self):
        import pickle

        from ai.genetic_gym import SAVE_PATH

        with open(SAVE_PATH, "rb") as file:
            ship_ai = pickle.load(file)

        self.start(ship_ai=ship_ai)


if __name__ == "__main__":
    Game().start()

    # Run trained AI visually
    # Game(visual=True).load_ai()

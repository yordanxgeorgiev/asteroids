"""Capture game state in a neural-network-friendly format."""

from __future__ import annotations

import dataclasses
import heapq
import math
from typing import TYPE_CHECKING, Optional

from constants import SCREEN_HEIGHT, SCREEN_WIDTH

if TYPE_CHECKING:
    from asteroid import Asteroid
    from player import Player
    from main import Game

# approximate maximum speeds.
MAX_SHIP_SPEED = 600.0
MAX_ASTEROID_SPEED = 300.0
MAX_RADIUS = 100.0


@dataclasses.dataclass
class GameState:
    """Compact neural-network input."""

    ship_position: tuple[float, float]
    ship_velocity: tuple[float, float]
    ship_direction: tuple[float, float]  # cos(theta), sin(theta)

    # distance,
    # cos(angle),
    # sin(angle),
    # rel_vx,
    # rel_vy,
    # radius
    nearest_asteroids: list[tuple[float, float, float, float, float, float]]


def get_game_state(game: "Game") -> Optional[GameState]:
    if not game.asteroids:
        return None

    ship = game.player

    rotation = math.radians(ship.rotation)

    ship_position = (
        ship.position.x / SCREEN_WIDTH,
        ship.position.y / SCREEN_HEIGHT,
    )

    ship_velocity = (
        ship.velocity.x / MAX_SHIP_SPEED,
        ship.velocity.y / MAX_SHIP_SPEED,
    )

    ship_direction = (
        math.cos(rotation),
        math.sin(rotation),
    )

    nearest = []

    for asteroid in get_nearest_asteroids(
        ship,
        game.asteroids.sprites(),
        count=5,
    ):
        rel = asteroid.position - ship.position

        distance = math.hypot(rel.x, rel.y)
        max_distance = math.hypot(SCREEN_WIDTH, SCREEN_HEIGHT)

        angle = math.atan2(rel.y, rel.x) - rotation

        rel_velocity = asteroid.velocity - ship.velocity

        nearest.append(
            (
                distance / max_distance,
                math.cos(angle),
                math.sin(angle),
                rel_velocity.x / MAX_ASTEROID_SPEED,
                rel_velocity.y / MAX_ASTEROID_SPEED,
                asteroid.radius / MAX_RADIUS,
            )
        )

    while len(nearest) < 5:
        nearest.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    return GameState(
        ship_position=ship_position,
        ship_velocity=ship_velocity,
        ship_direction=ship_direction,
        nearest_asteroids=nearest,
    )


def get_nearest_asteroids(
    ship: "Player",
    asteroids: list["Asteroid"],
    count: int = 5,
) -> list["Asteroid"]:
    return heapq.nsmallest(
        count,
        asteroids,
        key=lambda a: math.hypot(
            a.position.x - ship.position.x,
            a.position.y - ship.position.y,
        ),
    )

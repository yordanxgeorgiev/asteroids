"""A simple neural network implementation."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from game_state import GameState

NDArray = npt.NDArray[np.float64]


class Action(IntEnum):
    """Available ship controls."""

    THRUST = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    FIRE = 3


def relu(x: NDArray) -> NDArray:
    """ReLU activation function."""
    return np.maximum(0, x)


def sigmoid(x: NDArray) -> NDArray:
    """Sigmoid activation function."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def he_scale(dim: int) -> float:
    """He initialization scale for ReLU layers."""
    return np.sqrt(2.0 / dim)


def xavier_scale(dim: int) -> float:
    """Xavier initialization scale for sigmoid layers."""
    return np.sqrt(1.0 / dim)


class DenseLayer:
    """Fully connected neural network layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Callable[[NDArray], NDArray],
        weights: Optional[NDArray] = None,
        biases: Optional[NDArray] = None,
        initializer: str = "he",
    ):
        if weights is not None:
            self.weights = weights
        else:
            if initializer == "xavier":
                scale = xavier_scale(input_dim)
            else:
                scale = he_scale(input_dim)

            self.weights = (
                np.random.randn(input_dim, output_dim) * scale
            )

        self.biases = (
            biases
            if biases is not None
            else np.zeros(output_dim)
        )

        self.activation = activation

    def forward(self, inputs: NDArray) -> NDArray:
        values = np.dot(inputs, self.weights) + self.biases
        return self.activation(values)


class NeuralNetwork:
    """Simple feed-forward neural network."""

    ACTION_THRESHOLD = 0.5

    def __init__(self, *layers: DenseLayer):
        self.layers = list(layers)

    def predict(
        self,
        inputs: Union[NDArray, GameState],
    ) -> list[Action]:
        """Return all actions the network wants to perform."""

        if not isinstance(inputs, np.ndarray):
            inputs = self.get_inputs(inputs)

        for layer in self.layers:
            inputs = layer.forward(inputs)

        return [
            action
            for action, value in zip(Action, inputs)
            if value >= self.ACTION_THRESHOLD
        ]

    @staticmethod
    def get_inputs(game_state: GameState) -> NDArray:
        """
        Convert GameState into a flat neural network input vector.

        Layout:

        Ship:
            position x,y          (2)
            velocity x,y          (2)
            direction cos,sin     (2)

        Each asteroid:
            distance              (1)
            angle cos,sin         (2)
            relative velocity x,y (2)
            radius                (1)

        Total:
            6 + (5 * 6) = 36 inputs
        """

        inputs: list[float] = []

        inputs.extend(game_state.ship_position)
        inputs.extend(game_state.ship_velocity)
        inputs.extend(game_state.ship_direction)

        for asteroid in game_state.nearest_asteroids:
            inputs.extend(asteroid)

        return np.asarray(inputs, dtype=np.float64)
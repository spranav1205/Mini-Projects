Here's a README for your micromouse code:

---

# Micromouse Maze Solver

This project is a basic implementation of a maze-solving algorithm for a micromouse using flood-fill logic. The micromouse navigates through a grid maze to find the shortest path to a target destination.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Code Structure](#code-structure)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Future Improvements](#future-improvements)

## Overview

The micromouse is a small autonomous robot designed to navigate a maze and find the quickest path to the goal. This code simulates the core logic used by the micromouse, including flood-fill algorithm implementation and basic movement decision-making.

## Features

- **Flood-Fill Algorithm**: A classic maze-solving algorithm that assigns distance values to each cell in the grid based on their distance from the target.
- **Decision-Making**: The micromouse decides its next move based on the lowest distance value in the surrounding cells.
- **Obstacle Detection**: The micromouse detects walls and updates the grid accordingly.

## Code Structure

- **`main()`**: The entry point of the program, responsible for initializing the grid and running the flood-fill algorithm.
- **`initiate()`**: Initializes the grid with default values and obstacles.
- **`flood_fill(int x, int y)`**: Implements the flood-fill algorithm to calculate the shortest path.
- **`enqueue(int x, int y)` / `dequeue()`**: Implements a queue to handle the cells during the flood-fill process.
- **`best_choice()`**: Determines the best direction to move based on the grid values.
- **`isRight()`, `isLeft()`, `isForward()`**: Functions to check for obstacles in the micromouse's path and update the grid.
- **`setup()` and `loop()`**: Placeholder functions for implementing the micromouse's continuous operation, intended for an embedded environment.

## How It Works

1. **Grid Initialization**: The grid is initialized with walls (marked with a value of `69`) and empty cells (marked with `0`).
2. **Flood-Fill Algorithm**: The flood-fill algorithm is applied, starting from the target cell. Each cell is assigned a value representing the distance to the target.
3. **Pathfinding**: The micromouse checks the surrounding cells and moves towards the one with the lowest value, indicating the shortest path.
4. **Obstacle Handling**: As the micromouse moves, it detects walls and updates the grid to prevent moving into blocked cells.
5. **Completion**: The micromouse reaches the target and celebrates (future implementation).

## Getting Started

To run the code, simply compile and execute it in a C environment. This is a simulation of the logic, and actual micromouse hardware would require additional integration with sensors and motors.

```bash
gcc micromouse.c -o micromouse
./micromouse
```

## Future Improvements

- **Integrate with Hardware**: Add functions to interact with micromouse sensors and motors.
- **Enhanced Decision-Making**: Improve the decision-making logic to handle more complex mazes and tie-breaking when multiple paths are available.
- **Obstacle Detection Logic**: Implement actual wall detection logic based on sensor input.
- **Expand Maze Size**: Generalize the code to handle different maze sizes dynamically.

---

## License
[MIT](LICENSE) © Pranav Suryawanshi

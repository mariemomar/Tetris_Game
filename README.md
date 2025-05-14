# Tetris AI with Genetic Algorithm

![Tetris Gameplay Screenshot](screenshot.png) *(optional: add actual screenshot later)*

An implementation of Tetris where an AI player is trained using genetic algorithms to develop optimal playing strategies.

## Table of Contents
- [Features](#features)
- [Implementation Details](#implementation-details)
- [Requirements](#requirements)
- [Installation](#installation)


## Features

- **Complete Tetris Implementation**: Fully functional Tetris game with all standard mechanics
- **AI Player**: Genetic algorithm-based AI that learns to play Tetris
- **Multiple Evaluation Heuristics**: 9 different board metrics used for move evaluation
- **Training System**: Configurable genetic algorithm training pipeline
- **Visualization**: Real-time display of the AI's gameplay
- **Logging**: Detailed logs of the evolutionary process and performance metrics

## Implementation Details

The AI uses a genetic algorithm to evolve optimal weights for evaluating Tetris moves based on several heuristics:

1. **Board Metrics**:
   - Maximum height
   - Difference between maximum height
   - Minimum height
   - Number of full lines
   - Number of gaps

2. **Genetic Algorithm**:
   - Population size: 12 chromosomes
   - Generations: 10+
   - Selection: Top 50% elitism
   - Crossover: Uniform crossover
   - Mutation: Gaussian noise with 10% rate

3. **Move Evaluation**:
   - Tests all possible rotations and positions
   - Scores each potential move using weighted heuristics
   - Selects the highest scoring valid move

## Requirements

- Python 3.7+
- Pygame 2.0+
- NumPy (optional, for some calculations)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tetris-ai-genetic.git
   cd tetris-ai-genetic
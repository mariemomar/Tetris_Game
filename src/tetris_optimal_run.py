import random
import numpy as np
import json
from tetris_base import *
from tetris_ga_train import (
    simulate_game_for_chromosome,
    calculate_fitness,
    WEIGHT_KEYS,
)

OPTIMAL_RUN_PIECE_SEED = 42
NUM_PIECES_TEST = 600
OPTIMAL_WEIGHTS_FILE = "../reports/optimal_tetris_weights.json"

random.seed(OPTIMAL_RUN_PIECE_SEED)
np.random.seed(OPTIMAL_RUN_PIECE_SEED)


class OptimalPlayerLogic:
    def __init__(self, weights_dict):
        self.weights = weights_dict.copy()
        self.fitness = 0.0
        self.lines_cleared_in_game = 0
        self.pieces_played_in_game = 0
        self.score_in_game = 0

    def get_ordered_weights_values(self):
        return [self.weights[key] for key in WEIGHT_KEYS]


def main_optimal_execution():
    try:
        with open(OPTIMAL_WEIGHTS_FILE, "r") as f:
            loaded_optimal_weights = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(
            f"Error loading {OPTIMAL_WEIGHTS_FILE}: {e}. Ensure tetris_ga_train.py has run successfully."
        )
        return

    print("Successfully loaded optimal weights for the run:")
    for key, val in loaded_optimal_weights.items():
        print(f"  {key}: {val:.4f}")

    optimal_player = OptimalPlayerLogic(weights_dict=loaded_optimal_weights)

    lines_cleared, pieces_played, score = simulate_game_for_chromosome(
        optimal_player, NUM_PIECES_TEST
    )
    fitness = calculate_fitness(pieces_played, score)

    optimal_player.lines_cleared_in_game = lines_cleared
    optimal_player.pieces_played_in_game = pieces_played
    optimal_player.score_in_game = score
    optimal_player.fitness = fitness

    print("\n--- OPTIMAL RUN COMPLETE ---")
    print(f"Pieces Played: {pieces_played}/{NUM_PIECES_TEST}")
    print(f"Total Lines Cleared: {lines_cleared}")
    print(f"Total Score: {score}")
    print(f"Fitness: {fitness}")

    results_filename = "../reports/optimal_run_results.txt"
    try:
        with open(results_filename, "w") as f:
            f.write("--- OPTIMAL RUN RESULTS ---\n")
            f.write(f"Target Pieces: {NUM_PIECES_TEST}\n")
            f.write(f"Piece Seed Used: {OPTIMAL_RUN_PIECE_SEED}\n")
            f.write(f"Pieces Played: {pieces_played}\n")
            f.write(f"Total Lines Cleared: {lines_cleared}\n")
            f.write(f"Total Score: {score}\n")
            f.write(f"Fitness: {fitness}\n")
            f.write("\nWeights Used:\n")
            f.write(json.dumps(optimal_player.weights, indent=4) + "\n")
        print(f"Optimal run results saved to {results_filename}")
    except IOError:
        print(f"Error: Failed to write to {results_filename}.")


if __name__ == "__main__":
    MANUAL_GAME = False

    main_optimal_execution()

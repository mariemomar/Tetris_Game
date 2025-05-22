import random
import numpy as np
import json
from tetris_base import *

RANDOM_SEED_GA = 42
random.seed(RANDOM_SEED_GA)
np.random.seed(RANDOM_SEED_GA)


POPULATION_SIZE = 20  # >= 12
MUTATION_RATE = 0.1
GENERATIONS = 30  # >= 10
NUM_PIECES_TRAIN = 500  # 300-500
ELITISM_COUNT = 3


WEIGHT_KEYS = [
    "w_lines_cleared_move",  # Weight for lines cleared by the current single move
    "w_holes_total",  # Total number of holes
    "w_aggregate_height",  # Sum of all column heights
    "w_max_col_height",  # Max column height
    "w_bumpiness_total",  # Sum of height differences between adjacent columns
    "w_risk_near_top",  # Penalty for board getting too high
]

LOG_FILE = "../reports/tetris_ga_log.txt"
OPTIMAL_WEIGHTS_FILE = "../reports/optimal_tetris_weights.json"


class Chromosome:
    def __init__(self, weights_dict=None):
        self.weights = {}
        if weights_dict is None:
            for key in WEIGHT_KEYS:
                self.weights[key] = random.uniform(-2.0, 2.0)  # Initial range
        else:
            self.weights = weights_dict.copy()

        self.fitness = 0.0
        self.lines_cleared_in_game = 0
        self.pieces_played_in_game = 0
        self.score_in_game = 0

    def get_ordered_weights_values(self):
        return [self.weights[key] for key in WEIGHT_KEYS]

    def mutate(self, mutation_rate):
        for key in self.weights:
            if random.random() < mutation_rate:
                change = random.uniform(-0.3, 0.3)  # Mutation step size
                self.weights[key] += change
        return self


def calculate_score(lines_cleared_this_move):
    if lines_cleared_this_move == 1:
        return 40
    elif lines_cleared_this_move == 2:
        return 120
    elif lines_cleared_this_move == 3:
        return 300
    elif lines_cleared_this_move == 4:
        return 1200
    return 0


def crossover(parent1, parent2):
    child_weights = {}
    for key in WEIGHT_KEYS:
        if random.random() < 0.5:
            child_weights[key] = parent1.weights[key]
        else:
            child_weights[key] = parent2.weights[key]
    return Chromosome(weights_dict=child_weights)


def get_column_heights(board_col_major):
    heights = [0] * BOARDWIDTH
    for c in range(BOARDWIDTH):
        for r in range(BOARDHEIGHT):
            if board_col_major[c][r] != BLANK:
                heights[c] = BOARDHEIGHT - r
                break
    return heights


def count_holes(board_col_major):
    holes = 0
    for c in range(BOARDWIDTH):
        block_found_in_col = False
        for r in range(BOARDHEIGHT):
            if board_col_major[c][r] != BLANK:
                block_found_in_col = True
            elif block_found_in_col and board_col_major[c][r] == BLANK:
                holes += 1
    return holes


def get_risk_near_top(board_col_major):
    col_heights = get_column_heights(board_col_major)
    max_h = max(col_heights)
    if max_h >= BOARDHEIGHT - 4:
        return (max_h - (BOARDHEIGHT - 5)) * 2
    return 0


def evaluate_board_state(
    board_col_major, lines_cleared_this_move, ordered_weights_list
):
    col_heights = get_column_heights(board_col_major)

    aggregate_height = sum(col_heights)
    max_col_h = max(col_heights)

    bumpiness = 0
    if len(col_heights) > 1:
        for i in range(len(col_heights) - 1):
            bumpiness += abs(col_heights[i] - col_heights[i + 1])

    num_total_holes = count_holes(board_col_major)
    risk_val = get_risk_near_top(board_col_major)

    features = [
        lines_cleared_this_move,
        num_total_holes,
        aggregate_height,
        max_col_h,
        bumpiness,
        risk_val,
    ]

    score = sum(f * w for f, w in zip(features, ordered_weights_list))
    return score


def simulate_game_for_chromosome(chromosome, max_game_pieces):
    current_board_col_major = get_blank_board()
    total_lines_cleared_game = 0
    pieces_played_game = 0
    total_score_game = 0

    for _ in range(max_game_pieces):
        current_falling_piece = get_new_piece()

        if not is_valid_position(
            current_board_col_major, current_falling_piece, adj_X=0, adj_Y=0
        ):
            break

        best_move_score = float("inf")
        best_move_details = None

        for rot_idx in range(len(PIECES[current_falling_piece["shape"]])):
            piece_this_rotation = dict(current_falling_piece)
            piece_this_rotation["rotation"] = rot_idx

            for x_candidate_board_col in range(-TEMPLATEWIDTH + 1, BOARDWIDTH):
                piece_at_x = dict(piece_this_rotation)
                piece_at_x["x"] = x_candidate_board_col
                piece_at_x["y"] = -2

                sim_piece_to_drop = dict(piece_at_x)
                while is_valid_position(
                    current_board_col_major, sim_piece_to_drop, adj_Y=1
                ):
                    sim_piece_to_drop["y"] += 1

                if not is_valid_position(
                    current_board_col_major, sim_piece_to_drop, adj_Y=0
                ):
                    continue

                safe_to_add = True
                shape_template_for_check = PIECES[sim_piece_to_drop["shape"]][
                    sim_piece_to_drop["rotation"]
                ]
                for y_tpl_check in range(TEMPLATEHEIGHT):
                    for x_tpl_check in range(TEMPLATEWIDTH):
                        if shape_template_for_check[y_tpl_check][x_tpl_check] != BLANK:
                            board_x_final = x_tpl_check + sim_piece_to_drop["x"]
                            board_y_final = y_tpl_check + sim_piece_to_drop["y"]

                            if not (
                                0 <= board_x_final < BOARDWIDTH
                                and 0 <= board_y_final < BOARDHEIGHT
                            ):
                                safe_to_add = False
                                break
                    if not safe_to_add:
                        break

                if not safe_to_add:
                    continue

                temp_board_col_major = [
                    col_data[:] for col_data in current_board_col_major
                ]
                add_to_board(temp_board_col_major, sim_piece_to_drop)

                lines_cleared_by_this_specific_move = remove_complete_lines(
                    temp_board_col_major
                )

                current_evaluated_score = evaluate_board_state(
                    temp_board_col_major,
                    lines_cleared_by_this_specific_move,
                    chromosome.get_ordered_weights_values(),
                )

                if current_evaluated_score < best_move_score:
                    best_move_score = current_evaluated_score
                    best_move_details = (
                        sim_piece_to_drop,
                        temp_board_col_major,
                        lines_cleared_by_this_specific_move,
                    )

        if best_move_details is None:
            break

        _final_piece_state, current_board_col_major, lines_this_turn = best_move_details
        total_lines_cleared_game += lines_this_turn
        total_score_game += calculate_score(lines_this_turn)
        pieces_played_game += 1

    return total_lines_cleared_game, pieces_played_game, total_score_game


def calculate_fitness(pieces_played, score):
    return score + (pieces_played * 5)


def select_chromosomes(population, elitism_count, population_size, tournament_size):
    elite_chromosomes = [
        Chromosome(weights_dict=population[i].weights) for i in range(elitism_count)
    ]

    num_to_breed = population_size - len(elite_chromosomes)

    parent_pool = []
    for _ in range(num_to_breed * 2):
        contenders = random.sample(population, tournament_size)
        parent_pool.append(max(contenders, key=lambda c: c.fitness))

    return elite_chromosomes, parent_pool


def run_genetic_algorithm():
    population = [Chromosome() for _ in range(POPULATION_SIZE)]
    best_overall_chromosome = None
    best_overall_fitness = -float("inf")

    log_entries = []
    header = f"GA Settings: Pop={POPULATION_SIZE}, Gens={GENERATIONS}, Pieces/Game={NUM_PIECES_TRAIN}, MutationRate={MUTATION_RATE}, Elitism={ELITISM_COUNT}, GA_Seed={RANDOM_SEED_GA}\n"
    print(header)
    log_entries.append(header)
    log_entries.append(
        "Generation,Best Fitness (Raw),Second-Best Fitness (Raw),Lines Cleared (Best),Pieces Played (Best),Best Chromo Weights\n"
    )

    for gen_idx in range(GENERATIONS):
        print(f"\n=== Generation {gen_idx + 1}/{GENERATIONS} ===")

        for chromo_idx, chromo in enumerate(population):
            lines_cleared, pieces_played, score = simulate_game_for_chromosome(
                chromo, NUM_PIECES_TRAIN
            )
            chromo.fitness = calculate_fitness(pieces_played, score)
            chromo.lines_cleared_in_game = lines_cleared
            chromo.pieces_played_in_game = pieces_played
            chromo.score_in_game = score

        population.sort(key=lambda c: c.fitness, reverse=True)

        current_gen_best_chromosome = population[0]

        gen_summary = (
            f"  Best this Gen: Fitness={current_gen_best_chromosome.fitness:.0f}, "
            f"Lines={current_gen_best_chromosome.lines_cleared_in_game}, "
            f"Pieces={current_gen_best_chromosome.pieces_played_in_game}, "
            f"Score={current_gen_best_chromosome.score_in_game}"
        )
        print(gen_summary)

        log_line_content = (
            f"{gen_idx + 1},"
            f"{current_gen_best_chromosome.fitness:.2f},"
            f"{population[1].fitness:.2f},"
            f"{current_gen_best_chromosome.lines_cleared_in_game},"
            f"{current_gen_best_chromosome.pieces_played_in_game},"
            f"{current_gen_best_chromosome.score_in_game},"
            f'"{json.dumps(current_gen_best_chromosome.weights)}"'
        )
        log_entries.append(log_line_content + "\n")

        if current_gen_best_chromosome.fitness > best_overall_fitness:
            best_overall_fitness = current_gen_best_chromosome.fitness
            best_overall_chromosome = Chromosome(
                weights_dict=current_gen_best_chromosome.weights
            )
            best_overall_chromosome.fitness = current_gen_best_chromosome.fitness
            best_overall_chromosome.lines_cleared_in_game = (
                current_gen_best_chromosome.lines_cleared_in_game
            )
            best_overall_chromosome.pieces_played_in_game = (
                current_gen_best_chromosome.pieces_played_in_game
            )
            best_overall_chromosome.score_in_game = (
                current_gen_best_chromosome.score_in_game
            )

            print(f"  *** NEW OVERALL BEST FOUND! ***")

        # Selection
        elite_chromosomes, parent_pool = select_chromosomes(
            population, ELITISM_COUNT, POPULATION_SIZE, tournament_size=3
        )

        next_generation_population = elite_chromosomes

        num_to_breed = POPULATION_SIZE - len(elite_chromosomes)
        for i in range(num_to_breed):
            p1_idx = random.randrange(len(parent_pool))
            p2_idx = random.randrange(len(parent_pool))
            while p2_idx == p1_idx and len(parent_pool) > 1:
                p2_idx = random.randrange(len(parent_pool))
            p1 = parent_pool[p1_idx]
            p2 = parent_pool[p2_idx]

            child = crossover(p1, p2)
            child.mutate(MUTATION_RATE)
            next_generation_population.append(child)

        population = next_generation_population

    print("\n=== TRAINING COMPLETE ===")
    if best_overall_chromosome:
        print(
            f"Best Overall Chromosome Fitness (Raw): {best_overall_chromosome.fitness:.0f}"
        )
        print(
            f"  Lines Cleared in Game: {best_overall_chromosome.lines_cleared_in_game}"
        )
        print(
            f"  Pieces Played in Game: {best_overall_chromosome.pieces_played_in_game}"
        )
        print(f"  Score in Game: {best_overall_chromosome.score_in_game}")

        print("  Optimal Weights:")
        for k, v in best_overall_chromosome.weights.items():
            print(f"    {k}: {v:.4f}")

        final_log_summary = (
            f"\n=== BEST OVERALL CHROMOSOME ===\n"
            f"Fitness (Raw): {best_overall_chromosome.fitness:.0f}\n"
            f"Lines Cleared: {best_overall_chromosome.lines_cleared_in_game}\n"
            f"Pieces Played: {best_overall_chromosome.pieces_played_in_game}\n"
            f"Score: {best_overall_chromosome.score_in_game}\n"
            f"Weights: {json.dumps(best_overall_chromosome.weights, indent=4)}\n"
        )
        log_entries.append(final_log_summary)

        with open(OPTIMAL_WEIGHTS_FILE, "w") as f:
            json.dump(best_overall_chromosome.weights, f, indent=4)
        print(f"\nOptimal weights saved to {OPTIMAL_WEIGHTS_FILE}")

    with open(LOG_FILE, "w") as f:
        f.writelines(log_entries)
    print(f"Training log saved to {LOG_FILE}")


if __name__ == "__main__":
    MANUAL_GAME = False

    run_genetic_algorithm()

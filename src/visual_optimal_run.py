import random
import json
import pygame
import time
import tetris_base
from tetris_base import *
from tetris_ga_train import WEIGHT_KEYS, evaluate_board_state, calculate_score

# Constants
OPTIMAL_RUN_PIECE_SEED = 42
NUM_PIECES_TEST = 600
OPTIMAL_WEIGHTS_FILE = "../reports/optimal_tetris_weights.json"
FPS = 60
FALL_DELAY = 0


# Set random seed
random.seed(OPTIMAL_RUN_PIECE_SEED)

# Initialize Pygame
try:
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    BASICFONT = pygame.font.SysFont("arial", 18)
    pygame.display.set_caption("Tetris AI - Optimal Run")
except Exception as e:
    print(f"ERROR: Failed to initialize Pygame: {e}")
    exit(1)

# Set Pygame globals in tetris_base.py
tetris_base.DISPLAYSURF = DISPLAYSURF
tetris_base.FPSCLOCK = FPSCLOCK
tetris_base.BASICFONT = BASICFONT
tetris_base.BGCOLOR = BGCOLOR
tetris_base.BORDERCOLOR = BORDERCOLOR
tetris_base.TEXTCOLOR = TEXTCOLOR


class OptimalPlayerLogic:
    def __init__(self, weights_dict):
        self.weights = weights_dict.copy()

    def get_ordered_weights_values(self):
        return [self.weights[key] for key in WEIGHT_KEYS]


def simulate_game_for_chromosome(player_ai_logic, max_game_pieces):
    current_board_col_major = get_blank_board()
    total_lines_cleared_game = 0
    pieces_played_game = 0
    total_lines_score = 0
    next_piece = get_new_piece()

    print(
        f"Starting optimal run: Target pieces={max_game_pieces}, Piece Seed={OPTIMAL_RUN_PIECE_SEED}"
    )

    for piece_num in range(max_game_pieces):
        current_falling_piece = next_piece
        next_piece = get_new_piece()

        if not is_valid_position(
            current_board_col_major, current_falling_piece, adj_X=0, adj_Y=0
        ):
            print(
                f"Optimal Run: Game Over - New piece ({piece_num+1}) cannot be placed at spawn."
            )
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
                    player_ai_logic.get_ordered_weights_values(),
                )

                if current_evaluated_score < best_move_score:
                    best_move_score = current_evaluated_score
                    best_move_details = (
                        sim_piece_to_drop,
                        temp_board_col_major,
                        lines_cleared_by_this_specific_move,
                    )

        if best_move_details is None:
            print(
                f"Optimal Run: Game Over - No valid move found for piece {piece_num+1}."
            )
            break

        final_piece_state = best_move_details[0]  # Only need the piece position

        # Animate the piece falling
        falling_piece = dict(final_piece_state)
        falling_piece["y"] = -2  # Start from spawn
        while is_valid_position(current_board_col_major, falling_piece, adj_Y=1):
            DISPLAYSURF.fill(BGCOLOR)
            draw_board(current_board_col_major)
            draw_status(total_lines_score, level=1)
            draw_next_piece(next_piece)
            draw_piece(falling_piece)
            pygame.display.update()
            FPSCLOCK.tick(FPS)
            time.sleep(FALL_DELAY)
            falling_piece["y"] += 1

        # Final position (back up one step if invalid)
        if not is_valid_position(current_board_col_major, falling_piece, adj_Y=0):
            falling_piece["y"] -= 1

        # Place the piece on the board
        add_to_board(current_board_col_major, falling_piece)
        lines_this_turn = remove_complete_lines(current_board_col_major)
        total_lines_cleared_game += lines_this_turn
        total_lines_score += calculate_score(lines_this_turn)
        pieces_played_game += 1

        # Render final state
        DISPLAYSURF.fill(BGCOLOR)
        draw_board(current_board_col_major)
        draw_status(total_lines_score, level=1)
        draw_next_piece(next_piece)
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        time.sleep(FALL_DELAY)

        # Handle Pygame events (once per piece to reduce overhead)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return total_lines_cleared_game, pieces_played_game
            if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                pygame.quit()
                return total_lines_cleared_game, pieces_played_game

        if (piece_num + 1) % 100 == 0:
            print(
                f"  Optimal Run: Piece {piece_num+1} placed. Total Lines: {total_lines_cleared_game}"
            )

    print("\n--- OPTIMAL RUN COMPLETE ---")
    print(f"Pieces Played: {pieces_played_game}/{max_game_pieces}")
    print(f"Total Lines Cleared: {total_lines_cleared_game}")
    print(f"Total Lines Score: {total_lines_score}")

    results_filename = "optimal_run_results.txt"
    with open(results_filename, "w") as f:
        f.write("Successfully loaded optimal weights for the run:\n")
        for key, val in player_ai_logic.weights.items():
            f.write(f"  {key}: {val:.4f}\n")
        f.write(
            f"Starting optimal run: Target pieces={max_game_pieces}, Piece Seed={OPTIMAL_RUN_PIECE_SEED}\n"
        )
        for i in range(100, pieces_played_game + 1, 100):
            estimated_lines = (
                int(total_lines_cleared_game * (i / pieces_played_game))
                if pieces_played_game > 0
                else 0
            )
            f.write(
                f"  Optimal Run: Piece {i} placed. Total Lines: {estimated_lines}\n"
            )
        f.write("\n--- OPTIMAL RUN COMPLETE ---\n")
        f.write(f"Pieces Played: {pieces_played_game}/{max_game_pieces}\n")
        f.write(f"Total Lines Cleared: {total_lines_cleared_game}\n")
        f.write(f"Total Lines Score: {total_lines_score}\n")
    print(f"Optimal run results saved to {results_filename}")

    return total_lines_cleared_game, pieces_played_game


def main_optimal_execution():
    try:
        with open(OPTIMAL_WEIGHTS_FILE, "r") as f:
            loaded_optimal_weights = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Optimal weights file '{OPTIMAL_WEIGHTS_FILE}' not found.")
        print(
            "Ensure 'tetris_ga_train.py' has been run successfully to generate this file."
        )
        return
    except json.JSONDecodeError:
        print(
            f"ERROR: Could not decode JSON from '{OPTIMAL_WEIGHTS_FILE}'. File might be corrupted."
        )
        return

    print("Successfully loaded optimal weights for the run:")
    for key, val in loaded_optimal_weights.items():
        print(f"  {key}: {val:.4f}")

    optimal_player = OptimalPlayerLogic(weights_dict=loaded_optimal_weights)
    simulate_game_for_chromosome(optimal_player, NUM_PIECES_TEST)
    pygame.quit()


if __name__ == "__main__":
    MANUAL_GAME = False
    print(
        f"Running Optimal Test with MANUAL_GAME = {MANUAL_GAME} (from tetris_optimal_run.py context)"
    )
    main_optimal_execution()

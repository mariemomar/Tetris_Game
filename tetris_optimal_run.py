import random
import json
from tetris_base import *
from tetris_ga_train import evaluate_board_state, WEIGHT_KEYS

OPTIMAL_RUN_PIECE_SEED = 42

NUM_PIECES_TEST = 600
OPTIMAL_WEIGHTS_FILE = "optimal_tetris_weights.json"


random.seed(OPTIMAL_RUN_PIECE_SEED)


class OptimalPlayerLogic:
    def __init__(self, weights_dict):
        self.weights = weights_dict.copy()

    def get_ordered_weights_values(self):
        return [self.weights[key] for key in WEIGHT_KEYS]


def simulate_optimal_game_run(player_ai_logic, max_game_pieces):
    current_board_col_major = get_blank_board()
    total_lines_cleared_game = 0
    pieces_played_game = 0

    print(
        f"Starting optimal run: Target pieces={max_game_pieces}, Piece Seed={OPTIMAL_RUN_PIECE_SEED}"
    )

    for piece_num in range(max_game_pieces):
        current_falling_piece = get_new_piece()
        # Initial check if the new piece can even be placed at its default spawn location
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
                # Drop the piece
                while is_valid_position(
                    current_board_col_major, sim_piece_to_drop, adj_Y=1
                ):
                    sim_piece_to_drop["y"] += 1

                # Check if the final resting position is valid
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

        _final_piece_state, current_board_col_major, lines_this_turn = best_move_details
        total_lines_cleared_game += lines_this_turn
        pieces_played_game += 1

        if (piece_num + 1) % 100 == 0:
            print(
                f"  Optimal Run: Piece {piece_num+1} placed. Total Lines: {total_lines_cleared_game}"
            )

    print("\n--- OPTIMAL RUN COMPLETE ---")
    print(f"Pieces Played: {pieces_played_game}/{max_game_pieces}")
    print(f"Total Lines Cleared: {total_lines_cleared_game}")

    results_filename = "optimal_run_results.txt"
    with open(results_filename, "w") as f:
        f.write("--- OPTIMAL RUN RESULTS ---\n")
        f.write(f"Target Pieces: {max_game_pieces}\n")
        f.write(f"Piece Seed Used: {OPTIMAL_RUN_PIECE_SEED}\n")
        f.write(f"Pieces Played: {pieces_played_game}\n")
        f.write(f"Total Lines Cleared: {total_lines_cleared_game}\n")
        f.write("\nWeights Used:\n")
        f.write(json.dumps(player_ai_logic.weights, indent=4) + "\n")
    print(f"Optimal run results saved to {results_filename}")

    return total_lines_cleared_game, pieces_played_game


def main_optimal_execution():
    with open(OPTIMAL_WEIGHTS_FILE, "r") as f:
        loaded_optimal_weights = json.load(f)

    print("Successfully loaded optimal weights for the run:")
    for key, val in loaded_optimal_weights.items():
        print(f"  {key}: {val:.4f}")

    optimal_player = OptimalPlayerLogic(weights_dict=loaded_optimal_weights)
    simulate_optimal_game_run(optimal_player, NUM_PIECES_TEST)


if __name__ == "__main__":
    MANUAL_GAME = False
    print(
        f"Running Optimal Test with MANUAL_GAME = {MANUAL_GAME} (from tetris_optimal_run.py context)"
    )
    main_optimal_execution()

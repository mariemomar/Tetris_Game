from tetris_base import *
from main import *
from copy import deepcopy

def run_AI_game():
    # Setup variables
    board              = get_blank_board()
    last_fall_time     = time.time()
    score              = 0
    level, fall_freq   = calc_level_and_fall_freq(score)

    falling_piece      = get_new_piece()
    next_piece         = get_new_piece()

    while True:
        if falling_piece is None:
            falling_piece = next_piece
            next_piece    = get_new_piece()
            score += 1
            last_fall_time = time.time()

            if not is_valid_position(board, falling_piece):
                # GAME-OVER
                return

        check_quit()
        
        # === AI Decision Logic ===
        best_move = get_best_move(board, falling_piece, next_piece)
        
        # Apply the best move: rotation, left/right movement
        falling_piece['rotation'] = best_move['rotation']
        falling_piece['x']        = best_move['x']

        # Instantly drop to the target Y position
        while is_valid_position(board, falling_piece, adj_Y=1):
            falling_piece['y'] += 1

        # Piece has landed, add to board
        add_to_board(board, falling_piece)
        num_removed_lines = remove_complete_lines(board)

        # Bonus scoring
        score += [0, 40, 120, 300, 1200][num_removed_lines]

        level, fall_freq = calc_level_and_fall_freq(score)
        falling_piece    = None

        # === Draw Everything ===
        DISPLAYSURF.fill(BGCOLOR)
        draw_board(board)
        draw_status(score, level)
        draw_next_piece(next_piece)

        if falling_piece is not None:
            draw_piece(falling_piece)

        pygame.display.update()
        FPSCLOCK.tick(FPS)


def get_best_move(board, piece, next_piece):
    best_score = float('-inf')
    best_state = None
    
    for rotation in range(4):  # Try all 4 rotations
        new_piece = piece.copy()
        new_piece['rotation'] = rotation

        # Loop through all valid x-positions
        for x_pos in range(-5, 5):
            new_piece['x'] = x_pos
            new_piece['y'] = 0
            
            # Drop the piece until it lands
            while is_valid_position(board, new_piece, adj_Y=1):
                new_piece['y'] += 1
            
            # Evaluate the board state after placement
            temp_board = deepcopy(board)
            add_to_board(temp_board, new_piece)
            score = evaluate_board(temp_board)
            
            # Choose the best move
            if score > best_score:
                best_score = score
                best_state = {'x': x_pos, 'rotation': rotation}

    return best_state





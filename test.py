import random
import numpy as np
from collections import defaultdict
from tetris_base import *

class Chromosome:
    """Represents one set of weights for evaluating Tetris moves"""
    def __init__(self, weights=None):
        self.weights = weights or {
            'max_height': random.uniform(-5, 5),
            'lines_cleared': random.uniform(0, 10),  # Strong positive reward
            'new_holes': random.uniform(-10, 0),     # Strong negative penalty
            'new_blocking': random.uniform(-5, 0),
            'piece_contacts': random.uniform(-2, 2),
            'floor_contacts': random.uniform(-3, 0),
            'wall_contacts': random.uniform(-1, 1),
            'bumpiness': random.uniform(-8, 0),      # Highly penalize unevenness
            'aggregate_height': random.uniform(-5, 0)
        }
        self.fitness = 0  # Will store game score achieved with these weights

    def mutate(self, mutation_rate=0.15):
        """Randomly modify some weights"""
        for key in self.weights:
            if random.random() < mutation_rate:
                # Gaussian mutation (small changes more likely)
                self.weights[key] += np.random.normal(0, 0.5)
                # Keep weights in bounds
                self.weights[key] = max(-10, min(10, self.weights[key]))
        return self

def crossover(parent1, parent2):
    """Create child by blending parents' weights"""
    child_weights = {}
    for key in parent1.weights:
        # Blend with 50% probability from each parent
        if random.random() < 0.5:
            child_weights[key] = parent1.weights[key]
        else:
            child_weights[key] = parent2.weights[key]
    return Chromosome(child_weights)

class GeneticAlgorithm:
    def __init__(self, population_size=12):
        self.population = [Chromosome() for _ in range(population_size)]
        self.generation = 0
        self.best_chromosome = None

    def evolve(self, generations=10):
        """Run the evolutionary process"""
        for _ in range(generations):
            self.evaluate_population()
            self.select()
            self.repopulate()
            self.generation += 1
        return self.best_chromosome

    def evaluate_population(self):
        """Simulate games to calculate fitness"""
        for chromosome in self.population:
            # Run Tetris game using these weights
            chromosome.fitness = self.simulate_game(chromosome.weights)
            
        # Track best performer
        current_best = max(self.population, key=lambda x: x.fitness)
        if not self.best_chromosome or current_best.fitness > self.best_chromosome.fitness:
            self.best_chromosome = current_best

    def select(self, elite_ratio=0.3):
        """Keep top performers (elitism)"""
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        keep = int(len(self.population) * elite_ratio)
        self.population = self.population[:keep]

    def repopulate(self):
        """Create new generation through crossover/mutation"""
        new_population = self.population.copy()  # Keep elites
        
        while len(new_population) < len(self.population):
            # Tournament selection
            parent1, parent2 = random.sample(self.population, 2)
            child = crossover(parent1, parent2).mutate()
            new_population.append(child)
            
        self.population = new_population

    def simulate_game(self, weights, max_pieces=300):
        """Run Tetris with given weights and return score"""
        # Pseudocode - integrate with your actual Tetris game
        score = 0
        board = get_blank_board()
        
        for _ in range(max_pieces):
            piece = get_new_piece()
            next_piece = get_new_piece()
            
            # Find best move using current weights
            best_move = self.find_best_move(board, piece, next_piece, weights)
            execute_move(board, best_move)
            
            # Update score
            score += calculate_score(board)  
            
            if game_over(board):
                break
                
        return score

    def find_best_move(self, board, piece, next_piece, weights):
        """Evaluate all possible moves using the weights"""
        best_score = -float('inf')
        best_rotation = 0
        best_x = 0
        
        # Test all possible rotations and positions
        for rotation in range(len(PIECES[piece['shape']])):
            for x in range(BOARDWIDTH):
                # Get evaluation metrics for this move
                metrics = evaluate_position(board, piece, x, rotation)
                
                if not metrics['valid']:
                    continue
                    
                # Calculate weighted score
                move_score = 0
                for key, value in metrics.items():
                    if key in weights:
                        move_score += weights[key] * value
                
                # Prioritize survival (avoid game over)
                if metrics['will_lose']:
                    move_score -= 10000
                
                if move_score > best_score:
                    best_score = move_score
                    best_rotation = rotation
                    best_x = x
                    
        return {'x': best_x, 'rotation': best_rotation}

def evaluate_position(board, piece, x, rotation):
    """Simulate a move and return evaluation metrics"""
    test_piece = {
        'shape': piece['shape'],
        'rotation': rotation,
        'x': x,
        'y': piece['y']
    }
    
    # Simulate piece dropping
    while is_valid_position(board, test_piece, adj_Y=1):
        test_piece['y'] += 1
    
    # Create hypothetical board
    new_board = simulate_board(board.copy(), test_piece)
    
    return {
        'valid': True,
        'max_height': calculate_max_height(new_board),
        'lines_cleared': count_lines_cleared(new_board),
        'new_holes': count_new_holes(board, new_board),
        # ... other metrics ...
    }
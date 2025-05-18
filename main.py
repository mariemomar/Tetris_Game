from tetris_base import *
import numpy as np


MANUAL_GAME = False

if MANUAL_GAME == True:
    if __name__ == "__main__":
        main()


class chromosome:
    def __init__(self , weights):
        if weights == None:
            self.weights ={
                'max_height': random.uniform(-5, 5),
                'total_height' : random.uniform(-5 , 5) , 
                'bumpiness' : random.uniform(-5, 5),
                'holes' : random.uniform(-5,5),
                'future_risk' : random.uniform(-5,5),
            }
        else:
            self.weights = weights
        self.fitness = 0 

    def mutation(self , mutation_rate):
        for key in self.weight :
            if random.random() < mutation_rate :
                self.weights[key] += random.uniform(-1, 1)
        return self        

    
def crossover(parent1, parent2):
    keys = list(parent1.weights.keys())  
    split = random.randint(1, len(keys)-1)  
    
    child_weights = {}
    for i, key in enumerate(keys):
        if i < split: # the first part from parent 1 
            child_weights[key] = parent1.weights[key]
        else: # the second part from parent 2
            child_weights[key] = parent2.weights[key]
    
    return chromosome(weights=child_weights)



class GA_algorithm :
    def __init__(self , population_size , mutation_rate , generations , weights=None):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.weights = weights

    def creat_population(self , population_size) : 
        return [chromosome(weights=self.weights) for _ in range(population_size)]

    


def count_holes(board): # count holes which have block above it only 
    holes = 0
    for col in range(board.shape[1]):
        block_found = False
        for row in range(board.shape[0]):
            if board[row][col] != 0:
                block_found = True
            elif block_found:
                holes += 1
    return holes

def column_heights(board):
    heights = []
    for col in range(board.shape[1]):
        col_data = board[:, col]
        height = board.shape[0] - np.argmax(col_data[::-1]) if np.any(col_data) else 0
        heights.append(height)
    return heights


def future_risk(board):
    heights = column_heights(board)  # 25 
    sorted_heights = sorted(heights, reverse=True)
    risk = 0
    max_height = sorted_heights[0]
    if max_height >= 22 : 
        return abs((BOARDHEIGHT - max_height) - 4)   # we have 4 level of risk , 4 is game over 

def evaluate_board(board, contribution_factors): # the minimum score is better 
    heights = column_heights(board)
    max_height = max(heights)
    risk = future_risk(board)
    bumpiness = sum([abs(heights[i] - heights[i+1]) for i in range(len(heights)-1)])
    total_height = sum(heights)
    holes = count_holes(board)
    features = [holes, total_height, max_height, bumpiness , risk]

    score = sum([f * w for f, w in zip(features, contribution_factors)])
    return score


def selection(population):
    scored = [
        (evaluate_board(chromo.board, chromo.lines_cleared, chromo.contribution_factors), chromo)
        for chromo in population
    ]
    
    scored.sort(key=lambda x: x[0])  # x[0] is the score
    selected = [chromo for (score, chromo) in scored[:len(population)//2]]

    return selected
    



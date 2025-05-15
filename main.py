from tetris_base import *

MANUAL_GAME = False

if MANUAL_GAME == True:
    if __name__ == "__main__":
        main()


class chromosome:
    def __init__(self , weights):
        if weights == None:
            self.weights ={
                'max_height': random.uniform(-5, 5),
                'min_height': random.uniform(-5, 5),
                'difference_between_max_height' : random.uniform(-5, 5),
                'num_of_gaps' : random.uniform(-5,5),
                'score_gained' : random.uniform(-5,5),
                'lines_cleared': random.uniform(0, 4),
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

    

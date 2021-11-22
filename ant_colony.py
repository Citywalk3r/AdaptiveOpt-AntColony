import numpy as np
from random import shuffle
from tqdm import tqdm
np.set_printoptions(precision=2, threshold=None, edgeitems=None, linewidth=1000)

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    if leftSpan == 0:
        return rightMax
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.

    trasnalted_value = rightMin + (valueScaled * rightSpan)
    # print(f"{value} was translated to {trasnalted_value}")
    
    return trasnalted_value
    
class Ant_Colony:

    def __init__(self, is_debug) -> None:
        self.is_debug = is_debug
        self.pheromone_tbl = None
    
    def initialize_pheromone_tbl(self, nodes, initial_pheromone):

        size = len(nodes)
        pheromone_tbl = np.full((size, size), initial_pheromone)
        np.fill_diagonal(pheromone_tbl, 0, wrap=False)
        return pheromone_tbl
    
    def aco(self, m, T, r, a, b, eval_f, rng, init_pos_f, nodes, heuristic_f, q0, xi, elitism_strategy):
        """
        Ant colony optimization algorithm

        Parameters:
            m: number of ants
            T: number of iterations (generations)
            r: evaporation factor (0<r<1)
            a: relative importance of pheromone
            b: relative importance of heuristic
        """

        best_score_so_far = 1e15
        best_ant_evaluations = []
        self.best_ant = None

        random_solution = [i for i in range(1,21)]
        shuffle(random_solution)
        initial_pheromone = 5/eval_f(random_solution)

        self.pheromone_tbl = self.initialize_pheromone_tbl(nodes, initial_pheromone)

        if self.is_debug:
            print(f"Initial pheromone table:\n{self.pheromone_tbl}\nShape: {self.pheromone_tbl.shape}")
        
        for _ in tqdm(range(T)):

            initial_positions = init_pos_f(rng, m)
            ants = []
    
            for pos in initial_positions:
                ant = Ant(is_debug=self.is_debug, nodes=nodes, colony_obj=self, eval_f=eval_f)
                ant.set_initial_position(pos)
                ants.append(ant)

            pheromone_table = np.zeros((20, 20))

            for idx, ant in enumerate(ants):

                while ant.available_nodes:
                    ant.move(a,b, heuristic_f,q0)

                ant.eval()
                ant.calculate_delta_tau()

                if ant.evaluation < best_score_so_far:
                    best_score_so_far = ant.evaluation
                    self.best_ant = ant

                
                if self.is_debug:
                    print(f"~~Ant {idx} scored {ant.evaluation}.")
                    print(f"~~Ant {idx}'s solution: {ant.solution}.")
                    print(f"~~Ant {idx}'s moves taken: {ant.moves_taken}.")

                # print(f"{delta_t} amount of pheromone to be added")
   
                if elitism_strategy == "best_so_far":
                    for move in self.best_ant.moves_taken:
                        pheromone_table[move[0]-1][move[1]-1] += self.best_ant.delta_tau
                else:
                    for move in ant.moves_taken:
                        pheromone_table[move[0]-1][move[1]-1] += ant.delta_tau
                if xi:
                    for move in ant.moves_taken:
                        self.pheromone_tbl[move[0]-1][move[1]-1] = (1-xi)*self.pheromone_tbl[move[0]-1][move[1]-1] + xi*initial_pheromone

           
            # Offline update
            self.pheromone_tbl = np.add(r*self.pheromone_tbl, pheromone_table)

            if self.is_debug:
                print(f'The new pheromone table is: {self.pheromone_tbl}')
            
            best_ant_evaluations.append(best_score_so_far)

        print(f'The final pheromone table is: {self.pheromone_tbl}')

        return best_ant_evaluations, self.best_ant.solution
            
class Ant:

    def __init__(self, is_debug, nodes, colony_obj, eval_f) -> None:
        self.colony = colony_obj
        self.is_debug = is_debug
        self.eval_f = eval_f
        self.solution = []
        self.moves_taken = []
        self.available_nodes = nodes.copy()
    
    def set_initial_position(self, pos):
        self.solution.append(pos)
        self.available_nodes.remove(pos)
    
    def move(self,a,b, heuristic_f,q0):

        # initialization
        move_list = []
        probability_list= []
        denominator = 0

        for node in self.available_nodes:

            heur = heuristic_f(self.solution[-1]-1, node-1)
            p_node_not_normalized = self.colony.pheromone_tbl[self.solution[-1]-1][node-1]**a * (heur)**b
            denominator += self.colony.pheromone_tbl[self.solution[-1]-1][node-1]**a * (heur)**b
            
            probability_list.append(p_node_not_normalized)
            move_list.append(node)

        probability_list = [p / denominator for p in probability_list]
        # print(probability_list)

        if q0 and np.random.uniform(0,1) <= q0:
            move = move_list[np.where(probability_list == np.max(probability_list))[0][0]]
        else:   
            move = np.random.choice(move_list, 1, replace=False, p=probability_list)[0]
        
        self.moves_taken.append((self.solution[-1], move))
        self.available_nodes.remove(move)
        self.solution.append(move)

        if self.is_debug:
            print(f"The next move will be from {self.solution[-2]} to {move} with probability")
            print(f"The solution so far is {self.solution}")
            print(f"The available moves are: {self.available_nodes}")
        
    
    def deposit_pheromone(self, amount):
        pass

    def eval(self):
        self.evaluation = self.eval_f(self.solution)
    
    def calculate_delta_tau(self):
        self.delta_tau = 1/self.evaluation
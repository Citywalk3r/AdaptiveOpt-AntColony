import numpy as np
from random import shuffle

class Ant_Colony:

    def __init__(self, is_debug) -> None:
        self.is_debug = is_debug
        self.pheromone_tbl = None
    
    def initialize_pheromone_tbl(self, nodes):
        size = len(nodes)
        pheromone_tbl = np.full((size, size), 150/size**2)
        np.fill_diagonal(pheromone_tbl, 0, wrap=False)
        return pheromone_tbl
    
    def aco(self, m, T, r, a, b, eval_f, rng, init_pos_f, nodes, heuristic_f):
        """
        Ant colony optimization algorithm

        Parameters:
            m: number of ants
            T: number of iterations (generations)
            r: evaporation factor (0<r<1)
            a: relative importance of pheromone
            b: relative importance of heuristic
        """

        best_ant_evaluations = []
        best_score_so_far = 1e15
        self.best_ant = None
        self.pheromone_tbl = self.initialize_pheromone_tbl(nodes)
        if self.is_debug:
            print(f"Initial pheromone table:\n{self.pheromone_tbl}\nShape: {self.pheromone_tbl.shape}")
        
        for it in range(T):

            initial_positions = init_pos_f(rng, m)
            ants = []

            for pos in initial_positions:
                # shuffle(nodes)
                ant = Ant(is_debug=self.is_debug, nodes=nodes, colony_obj=self)
                ant.set_initial_position(pos)
                ants.append(ant)

            pheromone_tables = []

            for idx, ant in enumerate(ants):
                while ant.available_nodes:
                    ant.move(a,b, heuristic_f)
                ant_evaluation = eval_f(ant.solution)

                if ant_evaluation < best_score_so_far:
                    best_score_so_far = ant_evaluation
                    self.best_ant = ant
            
                size = len(nodes)
                temp_pheromone_tbl = np.zeros((size, size))

            
                for move in ant.moves_taken:
                    temp_pheromone_tbl[move[0]-1][move[1]-1] = 1e-6*ant_evaluation

                pheromone_tables.append(temp_pheromone_tbl.copy())

                if self.is_debug:
                    print(f"~~Ant {idx} scored {ant_evaluation}.")
                    print(f"~~Ant {idx}'s solution: {ant.solution}.")
                    print(f"~~Ant {idx}'s moves taken: {ant.moves_taken}.")

            total_pheromone_tbl = np.add.reduce(pheromone_tables)

            if self.is_debug:
                # print(f"Pheromone tables: {pheromone_tables}, shape: {len(pheromone_tables)}")
                print(f"Pheromone tables total: {total_pheromone_tbl}, shape: {total_pheromone_tbl.shape}")
            
            # Pheromone online update
            self.pheromone_tbl = np.add(r*self.pheromone_tbl, total_pheromone_tbl)

            if self.is_debug:
                print(f'The new pheromone table is: {self.pheromone_tbl}')
            
            best_ant_evaluations.append(best_score_so_far)

            # for ant in ants:
            #     print(f"Path taken: {ant.solution}")
        # print(f'The final pheromone table is: {self.pheromone_tbl}')
        # print(f"Best ant's score after {it+1} iterations: {best_score_so_far} ")

        

        return best_ant_evaluations, self.best_ant.solution
            
class Ant:

    def __init__(self, is_debug, nodes, colony_obj) -> None:
        self.colony = colony_obj
        self.is_debug = is_debug
        self.solution = []
        self.moves_taken = []
        self.available_nodes = nodes.copy()
        self.current_node = None
    
    def set_initial_position(self, pos):
        self.solution.append(pos)
        self.available_nodes.remove(pos)
        self.current_node = pos
    
    def move(self,a,b, heuristic_f):
        move_list = []
        denominator = 0
        for node in self.available_nodes:
            denominator += self.colony.pheromone_tbl[self.current_node-1][node-1]**a * heuristic_f(self.current_node-1, node-1)**b
        
        for node in self.available_nodes:
            p_node = (self.colony.pheromone_tbl[self.current_node-1][node-1]**a * heuristic_f(self.current_node-1, node-1)**b) / denominator
            move_list.append([self.current_node, node, p_node])
        
        # Sort the combined list based on the move probability
        sorted_moves = sorted(move_list, key=lambda x: x[2])

        move = sorted_moves[-1]
        if self.is_debug:
            print(f"The sorted move list is: {sorted_moves}")
            print(f"The next move will be from {move[0]} to {move[1]} with probability {move[2]}")
        
        # Move
        self.current_node = move[1]
        self.solution.append(move[1])
        self.available_nodes.remove(move[1])
        self.moves_taken.append((move[0], move[1]))

        if self.is_debug:
            print(f"The solution so far is {self.solution}")
            print(f"The available moves are: {self.available_nodes}")
        
    
    def deposit_pheromone(self, amount):
        pass
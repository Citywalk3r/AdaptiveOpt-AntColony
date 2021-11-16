import numpy as np

class Ant_Colony:

    def __init__(self, is_debug) -> None:
        self.is_debug = is_debug
        self.pheromone_tbl = None
    
    def initialize_pheromone_tbl(self, nodes):
        size = len(nodes)
        pheromone_tbl = np.full((size, size), 1/size**2)
        np.fill_diagonal(pheromone_tbl, 0, wrap=False)
        return pheromone_tbl
    
    def aco(self, m, T, r, a, b, eval_f, seed, init_pos_f, nodes, heuristic_f):
        """
        Ant colony optimization algorithm

        Parameters:
            m: number of ants
            T: number of iterations (generations)
            r: evaporation factor (0<r<1)
            a: relative importance of pheromone
            b: relative importance of heuristic
        """
        ants = []
        initial_positions = init_pos_f(seed, m)
        self.pheromone_tbl = self.initialize_pheromone_tbl(nodes)
        if self.is_debug:
            print(f"Initial pheromone table:\n{self.pheromone_tbl}\nShape: {self.pheromone_tbl.shape}")
            print(f"Initial positions for {m} ants: {initial_positions} ")

        for pos in initial_positions:
            ant = Ant(is_debug=self.is_debug, nodes=nodes, colony_obj=self)
            ant.set_initial_position(pos)
            ants.append(ant)
        
        if self.is_debug:
            for ant in ants:
                print(f"Ant path so far: {ant.path}\nAvailable nodes: {ant.available_nodes}")

        ants[0].move(a,b, heuristic_f)
class Ant:

    def __init__(self, is_debug, nodes, colony_obj) -> None:
        self.colony = colony_obj
        self.is_debug = is_debug
        self.path = []
        self.available_nodes = nodes.copy()
        self.from_node = None
        self.current_node = None
    
    def set_initial_position(self, pos):
        self.path.append(pos)
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
        self.path.append(move[1])
        self.from_node = move[0]

        if self.is_debug:
            print(f"The path so far is {self.path}")
        
    
    def deposit_pheromone(self, amount):
        pass
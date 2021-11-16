import numpy as np

class Ant_Colony:

    def __init__(self, is_debug) -> None:
        self.is_debug = is_debug
    
    def initialize_pheromone_tbl(self):
        pass
    
    def aco(self, m, T, r, a, b, eval_f, seed, init_pos_f, nodes):
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
        print(f"Initial positions for {m} ants: {initial_positions} ")

        for pos in initial_positions:
            ant = Ant(is_debug=self.is_debug, nodes=nodes)
            ant.set_initial_position(pos)
            ants.append(ant)
        
        for ant in ants:
            print(ant.path, ant.available_nodes)

class Ant:

    def __init__(self, is_debug, nodes) -> None:
        self.is_debug = is_debug
        self.path = []
        self.available_nodes = nodes.copy()
    
    def set_initial_position(self, pos):
        self.path.append(pos)
        self.available_nodes.remove(pos)
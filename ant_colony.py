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
    
    def aco(self, m, T, r, a, b, eval_f, seed, init_pos_f, nodes, heuristic):
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
        print(f"Initial pheromone table:\n{self.pheromone_tbl}\nShape: {self.pheromone_tbl.shape}")
        print(f"Initial positions for {m} ants: {initial_positions} ")

        for pos in initial_positions:
            ant = Ant(is_debug=self.is_debug, nodes=nodes)
            ant.set_initial_position(pos)
            ants.append(ant)
        
        for ant in ants:
            print(f"Ant path so far: {ant.path}\nAvailable nodes: {ant.available_nodes}")

        flow, normalized_flow = heuristic(6,0)
class Ant:

    def __init__(self, is_debug, nodes) -> None:
        self.is_debug = is_debug
        self.path = []
        self.available_nodes = nodes.copy()
    
    def set_initial_position(self, pos):
        self.path.append(pos)
        self.available_nodes.remove(pos)
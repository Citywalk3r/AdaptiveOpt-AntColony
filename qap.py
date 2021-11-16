from pathlib import Path
import numpy as np
from functools import reduce
from ant_colony import Ant_Colony
import matplotlib.pyplot as plt
import pandas as pd


def parse_data():
    distance_data_file = Path("dist_tbl.csv")
    try:
        distance_data_file.resolve(strict=True)
    except FileNotFoundError:
        print ("dist_tbl.csv not found. Please include the data file in the root folder. Aborting..\n")
        return
    else:
        flow_data_file = Path("flow_tbl.csv")
        try:
            flow_data_file.resolve(strict=True)
        except FileNotFoundError:
            print ("flow_tbl.csv not found. Please include the data file in the root folder. Aborting..\n")
            return
        else:
            distance = np.genfromtxt(distance_data_file, dtype=int, delimiter=',')
            flow = np.genfromtxt(flow_data_file, dtype=int, delimiter=',')
            return flow, distance
    
def generate_sq_tbl(currState):
    """Generates 20x20 matrix with one-hot encoding of the current state.
        If department 15 has index 2, element (15,1) = 1

        Example: [10,12,8,9,11,6,5,3,13,1,15,4,14,2,7,16,17,18,19,20]

        becomes

        [[0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]
        [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]]
    """

    x_t = np.zeros((20,20), dtype=int)
    for index in range(len(currState)):
        x_t[currState[index]-1][index] = 1
    return x_t

def NormalizeData(min, max, point):
    return (point - min) / (max - min)

class QAP:

    def __init__(self, is_debug):
        self.is_debug = is_debug
        self.flow, self.dist = parse_data()
        self.min_flow = np.min(self.flow)
        self.max_flow = np.max(self.flow)
        print(self.flow)

    def eval_func(self, currState):
        """https://en.wikipedia.org/wiki/Quadratic_assignment_problem
            score = trace(W * X * D * X_transp)
        """
        x_t = generate_sq_tbl(currState)
        score = np.trace(reduce(np.dot, [self.flow, x_t, self.dist, x_t.T]))
        # print(score)
        return score
    
    def generate_init_positions(self, seed, m):
        """
        Generates the initial ant position.
        """
        departments = list(range(1,21))
        rng = np.random.default_rng(seed)
        positions = [rng.choice(departments) for _ in range(m)]
        return positions
    

    
    def heuristic(self, last_node_visited, node_to_be_added):
        flow = self.flow[last_node_visited][node_to_be_added]
        normalized_flow = NormalizeData(self.min_flow, self.max_flow, flow)
        return flow, normalized_flow

    def solve_qap(self):
        """
        Solves the qap problem.
        """
        fig = plt.figure(figsize=(10, 5))
        AC = Ant_Colony(is_debug=self.is_debug)
        data = []

        headers = ['iterations', 'h', 776, 12, 234, 9238, 123556, 59933, 98232, 85732, 5432, 12291]
        seeds = [776]
        # seeds = [776, 12, 234, 9238, 123556, 59933, 98232, 85732, 5432, 12291]
        iterations_list=[1]
        n_ants_list=[5]
        a = 0.5
        b = 0.5
        r = 0.1

        for iterations in iterations_list:
            for idx, n_ants in enumerate(n_ants_list):
                # plt.subplot(1,len(n_ants_list),idx+1)
                # plt.title('ants = {:}, iterations = {:}'.format(n_ants, iterations))
                # plt.xlabel('iterations')
                # plt.ylabel('score')
                best_per_seed = []
                for seed in seeds:
                    best_list, best = AC.aco(m=n_ants, T=iterations, r=r, a=a, b=b, 
                                            eval_f=self.eval_func, seed=seed, init_pos_f=self.generate_init_positions,
                                            nodes=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], heuristic=self.heuristic)

        #             plt.plot(range(len(best_list)), best_list, label=str(seed))
        #             plt.legend()

        #             print("Best solution: ", best)
        #             best_per_seed.append(best)

        #         tmp = [iterations, n_ants]
        #         tmp.extend(best_per_seed)
        #         data.append(tmp)
        #     plt.show()
        # df= pd.DataFrame(data=data, columns= list(map(str, headers)))
        # print(df)
        # df.to_excel("../tabu_10_seeds_t11_best_so_far.xlsx")
       
        
if __name__ == "__main__":
    QAP = QAP(is_debug=False)

    # Global optimum: 2570
    QAP.solve_qap()
    # print(QAP.eval_func([10,12,8,9,11,6,5,3,13,1,15,4,14,2,7,16,17,18,19,20]))
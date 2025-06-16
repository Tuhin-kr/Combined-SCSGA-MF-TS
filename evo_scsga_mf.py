import numpy as np
import matplotlib.pyplot as plt

from pymoo.algorithms.soo.nonconvex.ga import GA

from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.repair.rounding import RoundingRepair

from pymoo.operators.mutation.pm import PM

from pymoo.core.problem import Problem
from pymoo.optimize import minimize

import funcs


# values will be passed from experiment file
n_agents, m_tasks, init_sol = None, None, None

#common variables
stop_con = ('n_gen', 500)
population_size = 100
def global_var():
    global init_sol
    init_sol = np.random.randint(0, m_tasks, size=(population_size, n_agents))





## Genetic Algorithm
def genetic_algo():
    # pymoo Problem Define
    problem = Problem(n_var = n_agents,n_obj = 1,n_constr = 0,
                      xl = 0,xu = m_tasks - 1,type_var = np.int32)
    # evaluation method
    problem._evaluate = lambda X, out, *args, **kwargs: out.update(F=funcs.main_fun(X))
    
    # Genetic Algo. Define
    algorithm = GA(
        pop_size = population_size,
        selection = RandomSelection(),
        crossover=SBX(eta=15, prob=0.9,repair=RoundingRepair()),
        mutation=PM(eta=20,repair=RoundingRepair()),
        mutation_prob=0.3,
        eliminate_duplicates=True,
        keep_parents=2,
        sampling = init_sol
    )
    
    # Run Genetic Algo. with Minimization
    res = minimize(problem,algorithm,termination=(stop_con),
                   seed=42,save_history=True)#,verbose=True
    
    # Results
    #best_solution = np.clip(np.round(res.X).astype(int), 0, m_tasks - 1)
    best_solution = res.X
    best_structure = funcs.decode_solution(best_solution) 
    best_sol_val = res.F[0]
    
    print("Best Solution Value (Genetic)=\n", best_sol_val )#, best_structure)
    
    #for verification of two values
    temp_final_val = funcs.col_struct_value(best_structure)
    temp_final_satisfy =  funcs.satisfy_value(best_structure)
    
    # store per iteration solutions and it's value
    fitness_history_val = [algo.pop.get("F").min() for algo in res.history]
                      #   [-algo.pop.get("F").min() for algo in res.history]
    best_idx = [np.argmin(algo.pop.get("F")) for algo in res.history]
    fitness_history_sol = [algo.pop.get("X")[best_idx] for algo in res.history]
    per_iter_sol = fitness_history_sol
    per_iter_val = [float(x) for x in fitness_history_val]
    #print("gen==\n",per_iter_sol)
    #print("gen==\n",per_iter_val)
    
    
    # Plot
    #plt.figure(figsize=(8, 5))
    plt.plot(fitness_history_val, marker='o', linestyle='-', color='blue')
    plt.title("SCSGA-MF-TS using Genetic Algo.")
    plt.xlabel("Generations")
    plt.ylabel("Best Fitness Values")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return best_sol_val, temp_final_val, temp_final_satisfy, best_structure, per_iter_val, per_iter_sol 

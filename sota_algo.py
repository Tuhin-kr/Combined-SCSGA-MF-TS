import copy
import time
import numpy as np
import itertools
from scipy.optimize import linear_sum_assignment

import funcs




# values will pass from experiment file
n_agents, m_tasks, init_sol = None, None, None

#common variables
stop_con = ('n_gen', 50)
population_size = 10
def global_var():
    global init_sol
    init_sol = np.random.randint(0, m_tasks, size=(population_size, n_agents))


### 1. Hill_Climb Algorithm:
def hill_climb(max_time=None, max_iter=None, CS=None):
    if max_time:
        start_time = time.time()
    elif max_iter:
        max_budget = max_iter
    else:
        print('At least one of max_time and max_iter must be set')
        return -1

    if CS is None:
        CS = np.random.randint(low=0, high=m_tasks, size=(n_agents,))
        CS[np.random.choice(n_agents, size=(m_tasks,), replace=False)] = np.arange(m_tasks)

    coalition_size = np.zeros((m_tasks,))
    for j in range(m_tasks):
        coalition_size[j] = (CS == j).sum()

    current_value = sum(funcs.col_main_fun(CS.copy()))
    CS_vals = [current_value]
    iter_count = 0

    while True:
        if max_time and (time.time() - start_time) >= max_time:
            break
        if max_iter and iter_count >= max_budget:
            break

        perm_agents = np.random.permutation(n_agents)
        for i in range(n_agents):
            if max_time and (time.time() - start_time) >= max_time:
                break

            ag = perm_agents[i]
            current_task = CS[ag]

            if coalition_size[current_task] == 1:
                continue

            for j in range(m_tasks):
                if max_time and (time.time() - start_time) >= max_time:
                    break

                if j == current_task:
                    continue

                # Tentative assignment
                CS[ag] = j
                new_value = sum(funcs.col_main_fun(CS))

                if new_value < current_value:
                    current_value = new_value
                    CS_vals.append(current_value)
                    coalition_size[current_task] -= 1
                    coalition_size[j] += 1
                else:
                    CS[ag] = current_task
                    CS_vals.append(current_value)

                # Time check after evaluation
                if max_time and (time.time() - start_time) >= max_time:
                    break

            # Another safeguard at agent-level
            if max_time and (time.time() - start_time) >= max_time:
                break

        iter_count += 1

    best_solution = np.clip(np.round(CS).astype(int), 0, m_tasks - 1)
    best_structure = funcs.decode_solution(best_solution)

    best_val = CS_vals
    print("Best Solution Value (hill) =\n", best_val[0])

    temp_final_val = funcs.col_struct_value(best_structure)
    temp_final_satisfy = funcs.satisfy_value(best_structure)

    per_iter_val = [float(x) for x in CS_vals.copy()]
    per_iter_sol = [CS.copy() for _ in per_iter_val]

    return best_val[0], temp_final_val, temp_final_satisfy, best_structure, per_iter_val, per_iter_sol



### 2. Simulated Annealing

def simulated_annealing(max_time=None, max_iter=None, CS=None):
    if max_time:
        max_budget = max_time
    elif max_iter:
        max_budget = max_iter
    else:
        print('At least one of max_time or max_iter must be set')
        return -1

    # Initialize coalition structure if not provided
    if CS is None:
        CS = np.random.randint(low=0, high=m_tasks, size=n_agents)
        # Ensure each task has at least one agent
        CS[np.random.choice(n_agents, size=m_tasks, replace=False)] = np.arange(m_tasks)

    # Maintain coalition sizes
    coalition_size = np.zeros(m_tasks, dtype=int)
    for j in range(m_tasks):
        coalition_size[j] = np.sum(CS == j)
    
    
    # Initial value of the current solution
    current_value = sum(funcs.col_main_fun(CS.copy()))
    CS_vals = [current_value]

    budget = 0
    while budget < max_budget:
        if max_time:
            start = time.time()

        # Randomly pick an agent and a different task
        rand_agent = np.random.randint(n_agents)
        current_task = CS[rand_agent]

        if coalition_size[current_task] == 1:  # Skip singleton coalition
            if max_time:
                budget += time.time() - start
                if budget > max_budget:
                    break
            continue

        other_tasks = np.delete(np.arange(m_tasks), current_task)
        rand_task = np.random.choice(other_tasks)

        # Tentatively assign and evaluate
        CS[rand_agent] = rand_task
        new_value = sum(funcs.col_main_fun(CS.copy()))
        

        if new_value < current_value:  # Accept move
            current_value = new_value
            coalition_size[current_task] -= 1
            coalition_size[rand_task] += 1
            CS_vals.append(current_value)
        else:  # Reject move
            CS[rand_agent] = current_task
            CS_vals.append(current_value)

        # Update budget
        if max_time:
            budget += time.time() - start
            if budget > max_budget:
                break
        elif max_iter:
            budget += 1
    
    # Final result
    best_solution = np.clip(np.round(CS).astype(int), 0, m_tasks - 1)
    best_structure = funcs.decode_solution(best_solution)
    
    best_val = CS_vals #or, funcs.main_fun(np.array([best_solution]))
    print("Best Solution Value (Simulated) =\n", best_val[0] )#, best_structure)
    #print("Best Solution (Greedy) =\n", best_structure)
    
    #for verification of two values
    temp_final_val = funcs.col_struct_value(best_structure)
    temp_final_satisfy =  funcs.satisfy_value(best_structure)
    
    #store per iteration Solution and it's value
    per_iter_val = [float(x) for x in CS_vals.copy()] # per iter value
    per_iter_sol = [CS.copy() for _ in per_iter_val] # per iter solution
    
    #print("Sim==\n",per_iter_sol)
    #print("Sim==\n",per_iter_val)
    
    return best_val[0], temp_final_val, temp_final_satisfy, best_structure, per_iter_val, per_iter_sol        
    #return CS_vals[-1], _, _, CS, CS_vals(#per iter Obj Value) 
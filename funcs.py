import numpy as np


# values will pass from experiment file
n_agents, m_tasks, agents, tasks, dist_mat = None, None, None, None, None
max_v_cs = None
dist_agent_agent = None

# for normalization of v(CS) & TC(CS,T)
def global_var():
    global max_v_cs
    max_v_cs = sum(-np.partition(-dist_mat.flatten(), n_agents-1)[:n_agents]) #sum of top n highest values
min_v_cs = 0      # min value is 0 for distnace



#fun to decode a gene into a coalition struct.
def decode_solution(any_sol):
        temp_sol = [[] for m in range(m_tasks)] 
        for i in range(n_agents): 
            if any_sol[i] < 0: # if -1 present, skip that agent (for sota)
                continue
            else:
                temp_sol[int(any_sol[i])].append(i)
        return temp_sol

#Two seperate functions
# coalition structure value
def col_struct_value(final_col_struct):
    all_coalition_value=[]
    for k in range (len(final_col_struct)):
        if len(final_col_struct[k])==0:
            all_coalition_value.append(0)
        else:
            all_coalition_value.append(sum(dist_mat[final_col_struct[k]][:,k]))
    return sum(all_coalition_value) 

# task satisfy value/count
def satisfy_value(final_col_struct):
    satisfy_mask = np.zeros(len(final_col_struct), dtype=bool)
    for j, coalition in enumerate(final_col_struct):
        if len(coalition) == 0:
            continue  # skip empty coalitions
        coalition_skills = np.sum(agents[coalition], axis=0)
        satisfy_mask[j] = np.all(coalition_skills >= tasks[j])
    return np.count_nonzero(satisfy_mask)

#Single objective (main function for optimization)
#for input as a gene space (encoded CS)
def main_fun(sol):
    results = []
    required_tasks = set(range(m_tasks))
    for x in sol:
        rounded_x = np.clip(np.round(x).astype(int), 0, m_tasks - 1) #convert to int value
        assigned_tasks = set(rounded_x)
        if not required_tasks.issubset(assigned_tasks):
            results.append(1e6)      # Penalty for invalid solution
            continue
        struct = decode_solution(rounded_x)
        v_cs = col_struct_value(struct)
        tc_cs = satisfy_value(struct)
        obj_val = (v_cs/max_v_cs) + (-(tc_cs/m_tasks))  # normalized & sum
        results.append(obj_val)        # pymoo minimizes
    return np.array(results)

# same main function to return coalition wise obj. val(v(cs)+T(CS))
def col_main_fun(sol):
    struct = decode_solution(sol)
    #calculate solution value
    all_coalition_value=np.zeros(len(struct))
    for k in range (len(struct)):
        if len(struct[k])==0:
            all_coalition_value[k]=0
        else:
            all_coalition_value[k]=(sum(dist_mat[struct[k]][:,k]))
    #calculate task-staisfaction count
    satisfy_mask = np.zeros(len(struct), dtype=bool)
    for coalition in range(len(struct)):
        if len(struct[coalition]) == 0:
            satisfy_mask[coalition] = False  # skip empty coalitions
        else:
            coalition_skills = np.sum(agents[satisfy_mask[coalition]], axis=0)
            satisfy_mask[coalition] = np.all(coalition_skills >= tasks[coalition])
    satisfy_mask = satisfy_mask.astype(int)
    
    obj_val = (all_coalition_value/max_v_cs) + (-(satisfy_mask/m_tasks))
    return obj_val


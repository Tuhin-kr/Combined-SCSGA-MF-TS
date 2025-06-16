import os
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import funcs
import evo_scsga_mf
import sota_algo
import dataset_generation



#reading the values of input varibale
confile = open('input_variable.txt', 'r')
con=[]
for line in confile:
    name, value = line.split("=")
    value = value.strip()
    con.append(int(value))
n_agents=con[0]
m_tasks=con[1]
features=con[2]
distribution_instances=con[3]
confile.close() 

# params. define
data_var = len(dataset_generation.variance)  # no. of data variances
noc = m_tasks
results_value = [] # to save the results loop wise
results_time = [] # to save the results loop wise
temp_result = [] # to save seperate value-count results
results_iter = [] # to save per iteration results

datanames = ['upd']
data_dir_path = '../dataset/'
result_dir_path = '../Results/'
if not os.path.exists(result_dir_path):
    os.makedirs(result_dir_path)
    


for t in range(len(datanames)):
    for var in range(data_var):
         for inst in range(distribution_instances):
    
                ##### Data Read
                main_data=np.load(data_dir_path+datanames[t]+str(inst)+'_v'+str(var)+'.npy')
                agents = main_data[0:n_agents] # from row 0:n_agents
                tasks = main_data[n_agents:]  #from row n_agents:
                
                #any fixed agent*task matrix construction
                dist_agent_task = cdist(agents, tasks, metric='euclidean')
                dist_mat = dist_agent_task
                dist_agent_agent = cdist(agents, agents, metric='euclidean')
                
                # Inject shared/global variables' value into algo
                funcs.n_agents = evo_scsga_mf.n_agents = sota_algo.n_agents = n_agents
                funcs.m_tasks = evo_scsga_mf.m_tasks = sota_algo.m_tasks = m_tasks
                funcs.agents = agents
                funcs.tasks = tasks
                funcs.dist_mat = dist_mat
                funcs.dist_agent_agent = dist_agent_agent
                # Use injected values within global_variable outside of funcion
                evo_scsga_mf.global_var()
                funcs.global_var()
                sota_algo.global_var()
                
                
                # each algorithm run
                #1. Genetic Algo.
                start=time.time()
                (gen_val, gen_temp_val, gen_temp_sat, _, 
                gen_iter_val, gen_iter_sol) = evo_scsga_mf.genetic_algo()  #main Gen. Algo
                end=time.time()
                gen_time = (end-start)
                
                de_val=de_temp_val=de_temp_sat=_= de_time =0  ## Dummy Values
                pso_val=pso_temp_val=pso_temp_sat=_=pso_time=0 ## Dummy Values
                brute_val = opt_time = opt_temp_val = opt_temp_sat = 0 ## Dummy Values
                greedy_val=gr_temp_val=gr_temp_sat=_=greedy_time=0 ## Dummy Values
                sol1 = count1 = 0 ## Dummy Values
                can_val = can_temp_val = can_temp_sat = _ = can_time=0 ## Dummy Values
                
                #2. Hill_Climb Algo.
                start=time.time()
                (hill_val, hill_temp_val, hill_temp_sat, _, 
                hill_iter_val, hill_iter_sol) =sota_algo.hill_climb(max_time=gen_time) #main Hill_climb algo
                end=time.time()
                hill_time = (end-start)
                #else:
                #hill_val=hill_time=hill_temp_val=hill_temp_sat= 0
                
                #3. Simulated Ann. Algo
                start=time.time()
                (sim_val, sim_temp_val, sim_temp_sat, _,
                sim_iter_val, sim_iter_sol) = sota_algo.simulated_annealing(max_time=gen_time) # main SA algo
                end=time.time()
                sim_time = (end-start)
                #else:
                #hill_val=hill_time=hill_temp_val=hill_temp_sat= 0

                
                # Result Construction (solution value)
                data_name = f"{datanames[t]}{inst}_v{var}" # Row name
                results_value.append({
                "Data": data_name, "GEN": gen_val, "DE": de_val, "PSO": pso_val,
                "Optimal": brute_val, "Greedy": greedy_val, "Hill_Climb":hill_val,
                "Sim_Annl":sim_val, "CAN":can_val
                 })
                # Result Construction (Execution time)
                results_time.append({
                "Data": data_name, "GEN": gen_time, "DE": de_time, "PSO": pso_time,
                "Optimal": opt_time, "Greedy": greedy_time, "Hill_Climb":hill_time,
                "Sim_Annl":sim_time, "CAN":can_time
                 })
                
                #For seperate value-count print
                # Result Construction (solution value)
                data_name = f"{datanames[t]}{inst}_v{var}" # Row name
                temp_result.append({
                "Data": data_name, "Metric": 'Total', "GEN": gen_val, "DE": de_val, 
                "PSO": pso_val,"Optimal": brute_val, "Greedy": greedy_val, 
                "Hill_Climb":hill_val, "Sim_Annl":sim_val, "CAN":can_val
                })
                temp_result.append({
                "Data": '', "Metric": 'Value', "GEN": gen_temp_val, "DE": de_temp_val, 
                "PSO": pso_temp_val,"Optimal": opt_temp_val, "Greedy": gr_temp_val, 
                "Hill_Climb":hill_temp_val, "Sim_Annl":sim_temp_val, "CAN":can_temp_val 
                })
                temp_result.append({
                "Data": '', "Metric": 'Count', "GEN": gen_temp_sat, "DE": de_temp_sat, 
                "PSO": pso_temp_sat,"Optimal": opt_temp_sat, "Greedy": gr_temp_sat, 
                "Hill_Climb":hill_temp_sat, "Sim_Annl":sim_temp_sat, "CAN":can_temp_sat
                })
                
                # save per iteration value and structure
                data = {"Data": data_name, "Gen_v": gen_iter_val,
                        "Hill_v": hill_iter_val, "Sim_v": sim_iter_val
                        }
                results_iter.append(data)
                
                
                
                print("-----"+f"{datanames[t]}{inst}_v{var}"+"--Completed-----\n")
                
                
# Result Save
df1 = pd.DataFrame(results_value)
df1.to_csv(result_dir_path+"Solution_value.csv", index=False)                
df2 = pd.DataFrame(results_time)
df2.to_csv(result_dir_path+"Execution_time.csv", index=False) 
df3 = pd.DataFrame(temp_result)
df3.to_csv(result_dir_path+"all_result.csv", index=False) 
df4 = pd.DataFrame(results_iter)
df4.to_csv(result_dir_path+"per_iter_result.csv", index=False)                  
                
                

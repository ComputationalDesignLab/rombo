import numpy as np
from scipy.io import loadmat
from aromad.test_problems.test_problems import EnvModelFunction
import matplotlib.pyplot as plt

data = loadmat('brusselator_results_v1.mat')
bo_data = data['BO_EI']
bologei_data = data['BO_LOGEI']
romboei_data = data['ROMBO_EI']
rombologei_data = data['ROMBO_LOGEI']

# Plotting the objective function history
plt.rcParams['font.size'] = 12
fig, ax = plt.subplots()
iter = np.arange(20,50,1)
ax.plot(iter, -bo_data['objectives'][0][0][0], 'bo-', label = 'BO+EI')
ax.plot(iter, -bologei_data['objectives'][0][0][0], 'mo-', label = 'BO+Log EI')
ax.plot(iter, -romboei_data['objectives'][0][0][0], 'ro-', label = 'ROMBO+EI')
ax.plot(iter, -rombologei_data['objectives'][0][0][0], 'go-', label = 'ROMBO+Log EI')
ax.grid()
ax.legend(loc = 'upper right')
ax.set_xlabel('Number of Samples')
ax.set_ylabel('Best Objective Function Value, $y^*$')
plt.tight_layout()
plt.savefig('./brusselatorpde_objective.pdf')
plt.show()

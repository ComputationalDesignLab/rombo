import numpy as np
from scipy.io import loadmat
from aromad.test_problems.test_problems import EnvModelFunction
import matplotlib.pyplot as plt

data = loadmat('env_model_results_256_q3.mat')
bo_data = data['BO']
romboei_data = data['ROMBO_EI']
rombologei_data = data['ROMBO_LOGEI']

# Plotting the objective function history
plt.rcParams['font.size'] = 14
fig, ax = plt.subplots()
iter = np.arange(10,40,3)
ax.semilogy(iter, -bo_data['objectives'][0][0][0], 'bo-', label = 'BO+EI')
ax.semilogy(iter, -romboei_data['objectives'][0][0][0], 'ro-', label = 'ROMBO+EI')
ax.semilogy(iter, -rombologei_data['objectives'][0][0][0], 'go-', label = 'ROMBO+Log EI')
ax.grid()
ax.legend()
ax.set_xlabel('Number of Samples')
ax.set_ylabel('Best Objective Function Value, $y^*$')
plt.tight_layout()
plt.show()

# Plotting the contours of the function
print(bo_data['design'])
x_list = [bo_data['design'][0][0][0][-1]]

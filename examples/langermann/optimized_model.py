import numpy as np
from scipy.io import loadmat, savemat
from hyperparameter_optimization import optimizeAutoencoder
import time
import warnings
warnings.filterwarnings("ignore")

n_samples = [50]
for size in n_samples:
    optimizer = optimizeAutoencoder(trainSamples=size, trainSize=0.8)
    t1 = time.time()
    optimizer.singleRunTrials(rounds = 30, numPerRound = 3)
    t2 = time.time()
    print("Time:", t2-t1)
    bestParams = optimizer.getBestParams()
    print("Best parameters:", bestParams)

    savemat("{}_hyperparameters.mat".format(size), bestParams)
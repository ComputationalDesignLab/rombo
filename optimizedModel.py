import numpy as np
from scipy.io import loadmat, savemat
from hyperparameterOptimization import optimizeAutoencoder
import warnings
warnings.filterwarnings("ignore")

n_samples = [50]
for size in n_samples:
    optimizer = optimizeAutoencoder(trainSamples=size, trainSize=0.8)
    optimizer.singleRunTrials(rounds = 10, numPerRound = 2)
    bestParams = optimizer.getBestParams()
    print("Best parameters:", bestParams)

    savemat("{}_hyperparameters.mat".format(size), bestParams)
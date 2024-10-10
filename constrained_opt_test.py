import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import warnings
warnings.filterwarnings('ignore')

d = 5

bounds = torch.stack([-torch.ones(d), torch.ones(d)])

train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(50, d)
train_Y = 1 - torch.norm(train_X, dim=-1, keepdim=True)

model = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

from botorch.acquisition import qExpectedImprovement
from botorch.sampling.stochastic_samplers import StochasticSampler

sampler = StochasticSampler(sample_shape=torch.Size([128]))
qEI = qExpectedImprovement(model, best_f=train_Y.max(), sampler=sampler)

N = 5
q = 1

from botorch.optim.initializers import initialize_q_batch_nonneg

# generate a large number of random q-batches
Xraw = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(100 * N, q, d)
Yraw = qEI(Xraw)  # evaluate the acquisition function on these q-batches

# apply the heuristic for sampling promising initial conditions
X = initialize_q_batch_nonneg(Xraw, Yraw, N)

# we'll want gradients for the input
X.requires_grad_(True)

print(X)

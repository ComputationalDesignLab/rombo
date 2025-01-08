# ROMBO: A composite Bayesian optimization framework for engineering design using nonintrusive reduced-order models

ROMBO is a optimization framework that utilizes a composite function formulation and nonlinear nonintrusive reduced order models. An autoencoder neural network is used to 
project high dimensional outputs into a latent space. The latent space is modeled using multi-task Gaussian process models that utilize a Kronecker structure or intrinsic model coregionalization (ICM) formulation. The framework uses a Monte Carlo expected improvement infill strategy to balance exploration of the design space with exploitation of the objective function. A linear POD method is also implemented using the same structure as ROMBO but using POD for dimensionality reduction and independent GP models for the latent space. A standard BO implementation is also provided for generating comparison data for the ROMBO framework. The framework is built utilizing PyTorch and associated libraries such as GPyTorch and BoTorch. Modular base classes have been provided for users to implement their own ROM architectures and utilize them within this framework.

img src="images/rombo.pdf"
     alt="ROMBO flowchart"
     style="float: left; margin-right: 10px;" />

# neural-stochastic-votality-model

A Neural Stochastic Volatility Model - based on the paper from Rui Luo, Weinan Zhang, Xiaojun Xu, and Jun Wang

Abstract of the paper

In this paper, we show that the recent integration of statistical models with deep recurrent neural networks provides a new way of formulating volatility (the degree of variation of time series) models that have been widely used in time series anal- ysis and prediction in finance. The model comprises a pair of complementary stochastic recurrent neural networks: the gen- erative network models the joint distribution of the stochas- tic volatility process; the inference network approximates the conditional distribution of the latent variables given the ob- servables. Our focus here is on the formulation of temporal dynamics of volatility over time under a stochastic recurrent neural network framework. Experiments on real-world stock price datasets demonstrate that the proposed model gener- ates a better volatility estimation and prediction that outper- forms mainstream methods, e.g., deterministic models such as GARCH and its variants, and stochastic models namely the MCMC-based model stochvol as well as the Gaussian process volatility model GPVol, on average negative log-likelihood.

Goal of the project

Implement a Neural Stochastic Volatility Model based on this paper.
 

References:
- This paper : A Neural Stochastic Volatility Model (Rui Luo, Weinan Zhang, Xiaojun Xu, Jun Wang) https://arxiv.org/pdf/1712.00504.pdf
- A Recurrent Latent Variable Model for Sequential Data (Junyoung Chung, Kyle Kastner, Laurent Dinh, Kratarth Goel, Aaron Courville, Yoshua Bengio) https://arxiv.org/abs/1506.02216
- Sequential Neural Models with Stochastic Layers (Marco Fraccaro, Søren Kaae Sønderby, Ulrich Paquet, Ole Winther) https://arxiv.org/pdf/1605.07571.pdf
- Autoregressive conditional het- eroscedasticity with estimates of the variance of united kingdom inflation (Robert F Engle) http://www.econ.uiuc.edu/~econ536/Papers/engle82.pdf
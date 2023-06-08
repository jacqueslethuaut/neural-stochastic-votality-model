# neural-stochastic-votality-model

A Neural Stochastic Volatility Model - based on the paper from Rui Luo, Weinan Zhang, Xiaojun Xu, and Jun Wang

Abstract of the paper

In this paper, we show that the recent integration of statistical models with deep recurrent neural networks provides a new way of formulating volatility (the degree of variation of time series) models that have been widely used in time series anal- ysis and prediction in finance. The model comprises a pair of complementary stochastic recurrent neural networks: the gen- erative network models the joint distribution of the stochas- tic volatility process; the inference network approximates the conditional distribution of the latent variables given the ob- servables. Our focus here is on the formulation of temporal dynamics of volatility over time under a stochastic recurrent neural network framework. Experiments on real-world stock price datasets demonstrate that the proposed model gener- ates a better volatility estimation and prediction that outper- forms mainstream methods, e.g., deterministic models such as GARCH and its variants, and stochastic models namely the MCMC-based model stochvol as well as the Gaussian process volatility model GPVol, on average negative log-likelihood.

Goal of the project

Implement a Neural Stochastic Volatility Model based on this paper.

Summarize

The paper by Luo et al. proposes a neural stochastic volatility model that integrates statistical models with deep recurrent neural networks to estimate and predict volatility in time series data.

Some key insights and lessons learned from the paper are:

- The neural stochastic volatility model can capture the complex dynamics and dependencies of volatility over time by using a pair of complementary stochastic recurrent neural networks: a generative network and an inference network.
- The model can handle both discrete and continuous observations, and can incorporate exogenous variables as additional inputs to improve the volatility estimation and prediction.
- The model outperforms several deterministic and stochastic models on real-world stock price datasets in terms of average negative log-likelihood, which measures how well the model fits the data.

A possible first implementation on the algorithm described in the paper is:

- Implement the generative network and the inference network using TensorFlow or PyTorch, following the architecture and parameters specified in the paper.
- Train the model on a stock price dataset using stochastic gradient descent with backpropagation through time, as described in the paper.
- Evaluate the model on a test set using average negative log-likelihood and other metrics such as mean squared error or mean absolute error.

Compare the results with other models such as GARCH, GPVol or StochVol.
 

References:
- This paper : A Neural Stochastic Volatility Model (Rui Luo, Weinan Zhang, Xiaojun Xu, Jun Wang) https://arxiv.org/pdf/1712.00504.pdf
- A Recurrent Latent Variable Model for Sequential Data (Junyoung Chung, Kyle Kastner, Laurent Dinh, Kratarth Goel, Aaron Courville, Yoshua Bengio) https://arxiv.org/abs/1506.02216
- Sequential Neural Models with Stochastic Layers (Marco Fraccaro, Søren Kaae Sønderby, Ulrich Paquet, Ole Winther) https://arxiv.org/pdf/1605.07571.pdf
- Autoregressive conditional het- eroscedasticity with estimates of the variance of united kingdom inflation (Robert F Engle) http://www.econ.uiuc.edu/~econ536/Papers/engle82.pdf
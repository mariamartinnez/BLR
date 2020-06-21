# BLR
Bayesian Logistic Regressor

# INTRO

Weights are assumed to follow a prior Laplacian distribution of zero mean, this way a feature will have to be informative in order to end with an associated weight different from zero:

<img src="https://render.githubusercontent.com/render/math?math=p(\bf{w})=\prod_{d=1}^{D}Laplace(w_d|0,b)= \prod_{d=1}^{D}\frac{1}{2b}exp\left(\frac{-|w_d|}{b}\right)">

Following the Logistic Regression model, likelihood is given by:

<img src="https://render.githubusercontent.com/render/math?math=p(\bf{Y}|\bf{X},\bf{w})=\prod_{n=1}^{N}p(Y_n|\bf{x}_n,\bf{w})=\prod_{n=1}^{N}\sigma(\bf{x}_n^T\bf{w})^{Y_n}(1-\sigma(\bf{x}_n^T\bf{w}))^{(1-Y_n)}">

where <img src="https://render.githubusercontent.com/render/math?math=\sigma"> refers to the Sigmod function.

This way, predictive distribution is given by:
<img src="https://render.githubusercontent.com/render/math?math=p(Y^*|\bf{x}^*)=\int p(Y^*|\bf{x}^*,\bf{w})p(\bf{w}|\bf{X},\bf{Y})d\bf{w}">

where the posterior distribution <img src="https://render.githubusercontent.com/render/math?math=p(\bf{w}|\bf{X},\bf{Y})"> , following Bayes' Theorem, can be expressed as
<img src="https://render.githubusercontent.com/render/math?math=p(\bf{w}|\bf{X},\bf{Y})=\frac{p(\bf{Y}|\bf{X},\bf{w})p(\bf{w})}{p(\bf{Y}|\bf{X})}">

However, this expression can't be used since the integral <img src="https://render.githubusercontent.com/render/math?math=p(\bf{Y}|\bf{X})=\int p(\bf{Y}|\bf{X},\bf{w})p(\bf{w})d\bf{w}"> is intractable. Thus, the posterior distribution is approximated by means of Variational Inference. 

This model assumes a Gaussian Variational Family, where mean vector and covariance matrix are learnt from data by means of Neural Networks. The usual approach is to use one Neural Network to learn the mean vector and another one to learn the diagonal of the covariance matrix, but in this case, in order to have a greater influence of the label, there are two Neural Networks to learn the mean vector and two to learn the covariance matrix, one for data with label 1 and another one for data with label 0. 

The posterior distribution is then approximated as a product of N Gaussians, i.e., it is computed as a contribution of every training sample, taking into account the value of the label. For each sample, a mean vector and a covariance matrix are computed by means of the corresponding Neural Networks. Then, the final expresion is given by  

<img src="https://render.githubusercontent.com/render/math?math=p(\bf{w}|\bf{X},\bf{Y}) \approx \prod_{n=1}^{N}N(\bf{\mu}_{\eta_1^1}(\bf{x}_n),\Sigma_{\eta_2^1}(\bf{x}_n))^{Y_n}N(\bf{\mu}_{\eta_1^0}(\bf{x}_n),\Sigma_{\eta_2^0}(\bf{x}_n))^{(1-Y_n)} = q(\bf{w}|\bf{X},\bf{Y})">

where <img src="https://render.githubusercontent.com/render/math?math=\eta"> refers to the parameters of the Neural Networks. Superscripts indicate the value of the label (0 or 1) and subscripts indicate wether the parameters correspond to the mean (1) or the covariance network (2). 

In order to find the weights that better explain training data, the Evidence Lower Bound (ELBO) is maximized during training, which is given by:

 <img src="https://render.githubusercontent.com/render/math?math=\int q(\bf{w}|\bf{X},\bf{Y})\log\frac{p(\bf{Y}|\bf{X},\bf{w})p(\bf{w})}{q(\bf{w}|\bf{X},\bf{Y})}d\bf{w} = \mathbb{E}_{q}\bigg[\log\frac{p(\bf{Y}|\bf{X},\bf{w})p(\bf{w})}{q(\bf{w}|\bf{X},\bf{Y})}\bigg] = \mathbb{E}_{q}\bigg[\log p(\bf{Y}|\bf{X},\bf{w})\bigg] %2B \mathbb{E}_{q}\bigg[\log p(\bf{w})\bigg] - \mathbb{E}_{q}\bigg[\log q(\bf{w}|\bf{X},\bf{Y})\bigg]">
 
 * <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}_{q}\bigg[\log p(\bf{Y}|\bf{X},\bf{w})\bigg]=\mathbb{E}_{q}\bigg[\sum_{n=1}^N Y_n\left(\log\sigma(\bf{x}_n^T\bf{w})\right) %2B (1-Y_n)\left(1-\log\sigma(\bf{x}_n^T\bf{w})\right) \bigg] = \mathbb{E}_{q}\bigg[\sum_{n=1}^N\big[-\log(1%2B\exp(-\bf{x}_n^T\bf{w})) %2B (1-Y_n)(-\bf{x}_n^T\bf{w})\big]\bigg]">
 
 *  <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}_{q}\bigg[\log p(\bf{w})\bigg] = \mathbb{E}_{q}\bigg[-\sum_{d=1}^{D}\big[\log(2b)%2B\frac{|w_d|}{b}\big]\bigg]">
 
 * <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}_{q}\bigg[\log q(\bf{w}|\bf{X},\bf{Y})\bigg] = \mathbb{E}_{q}\bigg[\log(N(\bf{\mu}_{\eta_1}(\bf{X},\bf{Y}), \Sigma_{\eta_2}(\bf{X},\bf{Y})))] = \mathbb{E}_{q}\bigg[-\frac{1}{2}\log(det(2\pi\Sigma_{\eta_2}(\bf{X},\bf{Y})))-\frac{1}{2}(\bf{w}-\bf{\mu}_{\eta_1}(\bf{X},\bf{Y}))^T\Sigma_{\eta_2}^{-1}(\bf{X},\bf{Y})(\bf{w}-\bf{\mu}_{\eta_1}(\bf{X},\bf{Y}))\bigg]">

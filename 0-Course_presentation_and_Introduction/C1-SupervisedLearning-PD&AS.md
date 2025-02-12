## 1. Statistical Learning, MESIO UPC-UB

# Chapter 1. Overview of Supervised learning. 

## 2. Introduction

## 3. Pedro Delicado (UPC)

With minor changes made by Alex Sanchez (UB)

1. Supervised and unsupervised learning
2. Statistical Learning Theory
3. The regression problem
4. The classification problem

## 4. Supervised and unsupervised learning

## 5. Statistical Learning Theory

3. The regression problem
4. The classification problem

## 6. Supervised learning

## 7. References:

Section 14.1 in Hastie, Tibshirani, and Friedman (2009)
Chapter 2 in James, Witten, Hastie, Tibshirani, and Taylor (2023)

## 8. Supervised Learning (the prediction problem)

- Let $(X, Y)$ be a r.v. with support $\mathcal{X} \times \mathcal{Y} \subseteq \mathbb{R}^{p} \times \mathbb{R}$.
- General supervised learning or prediction problem:
- Training sample: $S=\left\{\left(\boldsymbol{x}_{1}, y_{1}\right), \ldots,\left(\boldsymbol{x}_{n}, y_{n}\right)\right\}$, i.i.d. from $(\boldsymbol{X}, Y)$.
- The goal is to define a function (possibly depending on the sample) $h_{S}: \mathcal{X} \mapsto \mathcal{Y}$ such that for a new independent observation ( $\boldsymbol{x}_{n+1}, y_{n+1}$ ), from which we only know $\boldsymbol{x}_{n+1}$, it happens that

$$
\hat{y}_{n+1}=h_{S}\left(x_{n+1}\right) \text { is close to } y_{n+1} \text { (in some sense). }
$$

- Function $h_{S}$ is called generically prediction function (or classification function or regression function, depending on the case).
- We say that we have a problem of binary classification (or discrimination) when $\mathcal{Y}=\{0,1\}$ (you can also use $\mathcal{Y}=\{-1,1\}$ ).
- The problem of classification in $K$ classes arises when $\mathcal{Y}=\{1, \ldots, K\}$ (or $\left.\mathcal{Y}=\left\{\boldsymbol{y} \in\{0,1\}^{K}: \sum_{k=1}^{K} y_{k}=1\right\}\right)$.
- When $\mathcal{Y} \subseteq \mathbb{R}$ (or $\mathcal{Y}$ is an interval) we have a standard regression problem.


## 9. Supervised and unsupervised learning (I)

- Supervised learning (prediction problem):
- Regression: Predicting of a quantitative response.
- Classification (or discriminant analysis): Predicting a qualitative variable.
- Probabilistic model:
- Response variable $Y$.

Explanatory variables (features) $\boldsymbol{X}=\left(X_{1}, \ldots, X_{p}\right)$.

- Data $\left(\boldsymbol{x}_{i}=\left(x_{i 1}, \ldots, x_{i p}\right), y_{i}\right), i=1, \ldots, n$ i.i.d. from the random variable

$$
\left(\boldsymbol{X}=\left(X_{1}, \ldots, X_{p}\right), Y\right) \sim \operatorname{Pr}(\boldsymbol{X}, Y)
$$

- $\operatorname{Pr}(\boldsymbol{X}, \boldsymbol{Y})$ denotes the joint distribution of $\boldsymbol{X}$ and $Y$.
- When this joint distribution is continuous, $\operatorname{Pr}(\boldsymbol{X}, Y)$ is the joint probability density function.
- Main interest in the conditional distribution $\operatorname{Pr}(Y \mid \boldsymbol{X})$ and specifically in a conditional location parameter,

$$
\mu(\boldsymbol{x})=\underset{\mu}{\operatorname{argmin}} \mathbb{E}(L(Y, \mu) \mid \boldsymbol{X}=\boldsymbol{x}),
$$

where $L(y, \hat{y})$ is a loss function, measuring the error of predicting $y$ with $\hat{y}$.

- For quadratic loss, $L(y, \hat{y})=(y-\hat{y})^{2}, \mu(x)$ is the regression function:

$$
\mu(\boldsymbol{x})=\mathbb{E}(Y \mid \boldsymbol{X}=\boldsymbol{x})
$$

## 10. Supervised and unsupervised learning (II)

- Unsupervised learning: to learn relationships and structure from the observed data.
- Probabilistic model:
- Variables of interest: $\boldsymbol{X}=\left(X_{1}, \ldots, X_{p}\right)$.
- Data $\boldsymbol{x}_{i}=\left(x_{i 1}, \ldots, x_{i p}\right), i=1, \ldots, n$ i.i.d. from the random variable

$$
\boldsymbol{X}=\left(X_{1}, \ldots, X_{p}\right) \sim \operatorname{Pr}(\boldsymbol{X}) .
$$

- $\operatorname{Pr}(\boldsymbol{X})$ denotes the probability distribution of $\boldsymbol{X}$.
- When this distribution is continuous, $\operatorname{Pr}(\boldsymbol{X})$ is the probability density function of $\boldsymbol{X}$.
- Main interest: To infer properties of $\operatorname{Pr}(\boldsymbol{X})$.
- Specific problems in unsupervised learning:
- Estimating directly the density function $\operatorname{Pr}(\boldsymbol{x})$ :

Density estimation (histogram, kernel density estimation, Gaussian mixture models, ...)

- Detecting homogeneous subpopulations $C_{1}, \ldots, C_{k}$ :

$$
\operatorname{Pr}(x)=\sum_{j=1}^{k} \alpha_{j} \operatorname{Pr}\left(x \mid C_{j}\right), \alpha_{j} \geq 0, \sum_{j} \alpha_{j}=1
$$

Clustering (hierarchical clustering, $k$-means, Gaussian mixture models, ...)

- Finding low-dimensional hyper-planes or hyper-surfaces (manifolds) in $\mathbb{R}^{p}$ around which the probability $\operatorname{Pr}(\boldsymbol{x})$ is concentrated. Dimensionality reduction (PCA, MDS, principal curves, ISOMAP, manifold learning, ...)
- Proposing generative probabilistic models for $\boldsymbol{X}$, depending on low-dimensional unobservable random variables $\boldsymbol{F}$. Extraction of latent variables (Factor Analysis, ...)

1. Supervised and unsupervised learning
2. Statistical Learning Theory
3. The regression problem
4. The classification problem

## 11. Statistical Decision Theory

- We will write the prediction problem as a decision problem.
- Let $(\boldsymbol{X}, Y)$ be a r.v. with support $\mathcal{X} \times \mathcal{Y} \subseteq \mathbb{R}^{p} \times \mathbb{R}$.
- Prediction problem: To look for a prediction function $h: \mathcal{X} \mapsto \mathcal{Y}$ such that $h(\boldsymbol{X})$ is close to $Y$ in some sense.
- The (lack of) closeness between $h(\boldsymbol{X})$ and $Y$ is usually measured by a loss function $L(Y, h(\boldsymbol{X}))$.
- For instance, the squared error loss is $L(Y, h(\boldsymbol{X}))=(Y-h(\boldsymbol{X}))^{2}$.
- $L(Y, h(X))$ is a r.v., with expected value $\operatorname{EL}(h)=\mathbb{E}(L(Y, h(X)))$, called expected loss, that only depends on $h$.
- Decision problem: To find the prediction function $h: \mathcal{X} \mapsto \mathcal{Y}$ that minimizes the expected loss.


## 12. Bayes rule

Denote by $\operatorname{Pr}_{(\boldsymbol{X}, \boldsymbol{Y})}(\boldsymbol{x}, \boldsymbol{y})$ the joint probability distribution of $(\boldsymbol{X}, \boldsymbol{Y})$. Observe that, for any $h: \mathcal{X} \mapsto \mathcal{Y}$,

$$
\begin{gathered}
\mathrm{EL}(h)=\mathbb{E}(L(Y, h(\boldsymbol{X})))=\int_{\mathcal{X} \times \mathcal{Y}} L(y, h(\boldsymbol{x})) d \operatorname{Pr}_{(\boldsymbol{X}, \boldsymbol{Y})}(\boldsymbol{x}, \boldsymbol{y}) \\
=\int_{\mathcal{X}}\left(\int_{\mathcal{Y}} L(y, h(\boldsymbol{x})) d \operatorname{Pr}_{\boldsymbol{Y} \mid \boldsymbol{X}=\boldsymbol{x}}(\boldsymbol{y})\right) d \operatorname{Pr}_{\boldsymbol{X}}(\boldsymbol{x}) \\
=\int_{\mathcal{X}} \mathbb{E}(L(Y, h(\boldsymbol{x})) \mid \boldsymbol{X}=\boldsymbol{x}) d \operatorname{Pr}_{\boldsymbol{X}}(\boldsymbol{x}) \\
\geq \int_{\mathcal{X}} \min _{y \in \mathcal{Y}} \mathbb{E}(L(Y, y) \mid \boldsymbol{X}=\boldsymbol{x}) d \operatorname{Pr}_{\boldsymbol{X}}(\boldsymbol{x})=\mathrm{EL}\left(h_{B}\right) .
\end{gathered}
$$

It follows that the optimal prediction function is the Bayes rule

$$
h_{B}(\boldsymbol{x})=\arg \min _{y \in \mathcal{Y}} \mathbb{E}(L(Y, y) \mid \boldsymbol{X}=\boldsymbol{x}) .
$$

1. Supervised and unsupervised learning
2. Statistical Learning Theory
3. The regression problem
4. The classification problem

## 13. The regression problem

- Let $(\boldsymbol{X}, Y)$ be a $(p+1)$-dimensional random variable, with $Y \in \mathbb{R}$ (or in a subset of $\mathbb{R}$ ).
- The regression problem: To predict $Y$ from known values of $\boldsymbol{X}$.
- The most common and convenient loss function is the squared error loss: $L(Y, h(\boldsymbol{X}))=(Y-h(\boldsymbol{X}))^{2}$.
- The expected loss is known as Prediction Mean Squared Error, (PMSE):

$$
\operatorname{PMSE}(h)=\mathbb{E}\left((Y-h(\boldsymbol{X}))^{2}\right) .
$$

- The Bayes rule in this case is

$$
h_{B}(\boldsymbol{x})=\arg \min _{y \in \mathcal{Y}} \mathbb{E}\left((Y-y)^{2} \mid \boldsymbol{X}=\boldsymbol{x}\right) .
$$

- Bayes rule: $h_{B}(x)=\arg \min _{y \in \mathcal{Y}} \mathbb{E}\left((Y-y)^{2} \mid \boldsymbol{X}=\boldsymbol{x}\right)$.
- Observe that, for any $y \in \mathcal{Y}, \mathbb{E}\left((Y-y)^{2} \mid \boldsymbol{X}=\boldsymbol{x}\right)$

$$
\begin{aligned}
= & \mathbb{E}\left(((Y-\mathbb{E}(Y \mid \boldsymbol{X}=\boldsymbol{x}))+(\mathbb{E}(Y \mid \boldsymbol{X}=\boldsymbol{x})-\boldsymbol{y}))^{2} \mid \boldsymbol{X}=\boldsymbol{x}\right) \\
= & \mathbb{E}\left((Y-\mathbb{E}(Y \mid \boldsymbol{X}=\boldsymbol{x}))^{2} \mid \boldsymbol{X}=\boldsymbol{x}\right)+\underbrace{(\mathbb{E}(Y \mid \boldsymbol{X}=\boldsymbol{x})-\boldsymbol{y}))^{2}}_{\geq 0} \\
& +2(\mathbb{E}(Y \mid \boldsymbol{X}=\boldsymbol{x})-y) \mathbb{E}(Y-\mathbb{E}(Y \mid \boldsymbol{X}=\boldsymbol{x}) \mid \boldsymbol{X}=\boldsymbol{x}) \\
& \geq \mathbb{E}\left((Y-\mathbb{E}(Y \mid \boldsymbol{X}=\boldsymbol{x}))^{2} \mid \boldsymbol{X}=\boldsymbol{x}\right)
\end{aligned}
$$

- The Bayes rule here is the conditional expectation of $Y$ given $\boldsymbol{X}=\boldsymbol{x}$,

$$
h_{B}(x)=\mathbb{E}(Y \mid X=x),
$$

also known as regression function of $Y$ over $x$ and usually denoted by

$$
m(\boldsymbol{x})=\mathbb{E}(Y \mid \boldsymbol{X}=\boldsymbol{x}) .
$$

- Parametric regression models assume that $m(x)$ is known except for a finite number of unknown parameters,

$$
m(x) \equiv m(x ; \theta), \theta \in \Theta \subseteq \mathbb{R}^{q},
$$

- For instance, the multiple linear regression model postulates that $m(\boldsymbol{x})=\beta_{0}+\boldsymbol{x}^{\boldsymbol{\top}} \boldsymbol{\beta}_{1}$, with unknown parameters $\beta_{0} \in \mathbb{R}, \boldsymbol{\beta}_{1} \in \mathbb{R}^{\boldsymbol{p}}$.
- The training sample, $S=\left\{\left(x_{1}, y_{1}\right), \ldots,\left(x_{n}, y_{n}\right)\right\}$, i.i.d. from $(\boldsymbol{X}, Y)$, is used to estimate the parameter $\theta$.
- In this case $h_{S}(\boldsymbol{x})=m(\boldsymbol{x} ; \hat{\theta})$, where $\hat{\theta}=\hat{\theta}(S)$ is the estimation of $\theta$ from sample $S$.


## 14. Least squares estimation

- In this context the usual way to estimate $\theta$ is by least squares (LS):

$$
\hat{\theta}=\arg \min _{\theta \in \Theta} \sum_{i=1}^{n}\left(y_{i}-m\left(\boldsymbol{x}_{i} ; \theta\right)\right)^{2}
$$

- This is equivalent to the maximum likelihood estimation of $\theta$ when $(X, Y)$ is assumed to have a joint normal distribution. In this case:
- The regression function $m(\boldsymbol{x})$ is linear in $\boldsymbol{x}$.
- It is equivalent to state the model as

$$
Y=m(\boldsymbol{X})+\varepsilon,
$$

where $\varepsilon$ is an additive noise normally distributed with zero mean and independent from $\boldsymbol{X}$, also normally distributed.

## 15. Least squares estimation (II)

- Observe that the LS estimator $\hat{\theta}$ minimizes the prediction error in the training sample (Residual Sum of Squares: $\operatorname{RSS}(\theta)=\sum_{i=1}^{n}\left(y_{i}-m\left(\boldsymbol{x}_{i} ; \theta\right)\right)^{2}$ ), which takes its minimum value

$$
\overline{\mathrm{err}}=\operatorname{RSS}(\hat{\theta})=\sum_{i=1}^{n}\left(y_{i}-m\left(x_{i} ; \hat{\theta}\right)\right)^{2}
$$

- This quantity $\overline{\text { err }}$ is known as the training error or the apparent error.
- Usually it is an optimistic estimation of the Prediction Mean Squared Error (PMSE) in an observation of $\left(\boldsymbol{X}_{n+1}, Y_{n+1}\right)$ independent from the training sample,

$$
\operatorname{PMSE}(\theta)=\mathbb{E}\left(\left(Y_{n+1}-m\left(\boldsymbol{x}_{i} ; \theta\right)\right)^{2}\right),
$$

mainly when the parametric family $m(\boldsymbol{x} ; \theta), \theta \in \Theta \subseteq \mathbb{R}^{q}$, is too flexible:

$$
\overline{\operatorname{err}}<\operatorname{PMSE}(\hat{\theta}) \neq \min _{\theta \in \mathbb{R}^{q}} \operatorname{PMSE}(\theta)
$$

- This is the case in non-parametric regression and in many machine learning algorithms. (Example: $k$-nearest neighbors regression, where the tuning parameter is $k$ ).
- We will talk later in the course about cross-validation and tuning parameters.


## 16. Example: k nearest-neighbors

- The $k$ nearest-neighbor estimator of $m(\boldsymbol{t})=E(Y \mid \boldsymbol{X}=\boldsymbol{t})$ is defined as

$$
\hat{m}(\boldsymbol{t})=\frac{1}{\left|N_{k}(\boldsymbol{t})\right|} \sum_{i \in N_{k}(\boldsymbol{t})} y_{i},
$$

where $N_{k}(\boldsymbol{t})$ is the neighborhood of $\boldsymbol{t}$ defined by the $k$ closest points $\boldsymbol{x}_{i}$ in the training sample.

- Closeness is defined according to a previously chosen distance measure $d(\boldsymbol{t}, \boldsymbol{x})$, for instance, the Euclidean distance.


## 17. $k$-nn regression, in R

```
knn_regr<- function(x, y, t=NULL, k=3, dist.method = "euclidean"){
    nx <- length(y)
    if (is.null(t)){
        t<- as.matrix(x)
    } else {
        t<-as.matrix(t)
    }
    nt <- dim(t)[1]
    Dtx <- as.matrix( dist(rbind(t,as.matrix(x)),
    method = dist.method) )
    Dtx <- Dtx[1:nt,nt+(1:nx)]
    mt <- numeric(nt)
    for (j in 1:nt){
        d_t_x <- Dtx[j,]
        d_t_x_k <- sort(d_t_x,partial=k)[k]
        N_t_k <- unname( which(d_t_x <= d_t_x_k) )
        mt[j]=mean(y[N_t_k])
    }
    return(mt)
}
```


## 18. Example of $k-n n$ regression

```
n <- 200; sde <- .05
x <- sort(2*runif(n)-1)
mx <- 1-x^2; e <- rnorm(n,0,sde)
y <- mx + e
plot(x,y,xlim=c(-1,1),ylim=c(-3*sde, 1+3*sde), col=8)
lines(x,mx,col=4)
t<- seq(-1,1,by=.05)
k <- n/20
hat_mt <- knn_regr(x, y, t=t, k=k)
lines(t,hat_mt, col=2, lwd=2)
title(main=paste0("k-nn regression estimator, k=",k))
```

$\mathrm{k}-\mathrm{nn}$ regression estimator, $\mathrm{k}=10$
![](https://cdn.mathpix.com/cropped/2025_02_08_69f53f20c21a4f60a057g-21.jpg?height=294&width=568&top_left_y=623&top_left_x=330)

## 19. Practice:

## 20. Follow the Rmd file knn_regr.Rmd.

1. Supervised and unsupervised learning
2. Statistical Learning Theory
3. The regression problem
4. The classification problem

## 21. The classification problem

- Let $(\boldsymbol{X}, Y)$ be a r.v. with support $\mathcal{X} \times \mathcal{Y} \subseteq \mathbb{R}^{p} \times\{1, \ldots, K\}$. We want to predict $Y$ from observed values of $\boldsymbol{X}$.
- The loss function in this case can be represented by a $K \times K$ matrix $\boldsymbol{L}$, that will be zero on the diagonal and nonnegative elsewhere.
- The element $(j, k)$ of $\boldsymbol{L}$ is $L(j, k)$ : the price paid for classifying in class $k$ an observation belonging to class $j$.


## 22. The zero-one loss function

- Most often the zero-one loss function is used, where all misclassifications are charged a single unit.
- With the 0-1 loss function the Bayes rule is

$$
\begin{gathered}
h_{B}(\boldsymbol{x})=\arg \min _{y \in \mathcal{Y}} \mathbb{E}\left(L_{0-1}(Y, y) \mid \boldsymbol{X}=\boldsymbol{x}\right) \\
=\arg \min _{k \in\{1, \ldots, K\}} \sum_{j=1}^{K} L_{0-1}(j, k) \operatorname{Pr}(Y=j \mid \boldsymbol{X}=\boldsymbol{x}) \\
=\arg \min _{k \in\{1, \ldots, K\}}(1-\operatorname{Pr}(Y=k \mid \boldsymbol{X}=\boldsymbol{x}))=\arg \max _{k \in\{1, \ldots, K\}} \operatorname{Pr}(Y=k \mid \boldsymbol{X}=\boldsymbol{x}) .
\end{gathered}
$$

- In this context the Bayes rule is known as the Bayes classifier, and says that we classify to the most probable class, conditional to the observed value $\boldsymbol{x}$ of $\boldsymbol{X}$.


## 23. The problem of binary classification

- Consider the binary classification problem: $\mathcal{Y}=\{0,1\}$. Then
$(Y \mid \boldsymbol{X}=\boldsymbol{x}) \sim \operatorname{Bernoulli}(p=p(\boldsymbol{x})=\operatorname{Pr}(Y=1 \mid \boldsymbol{X}=\boldsymbol{x})=\mathbb{E}(Y \mid \boldsymbol{X}=\boldsymbol{x}))$.
- The Bayes classifier is

$$
h_{B}(x)=\left\{\begin{array}{lll}
1 & \text { if } & p(x) \geq 1 / 2 \\
0 & \text { if } & p(x)<1 / 2
\end{array}\right.
$$

- As $p(x)$ is unknown, we use a training sample to estimate it.
- Let $\left(x_{1}, y_{1}\right), \ldots,\left(x_{n}, y_{n}\right)$ be $n$ independent realizations of $(X, Y)$.
- Given an estimation $\hat{p}(\boldsymbol{x})$ of the regression function $p(\boldsymbol{x})$, the estimated version of the Bayes classifier is

$$
h_{S}\left(x_{n+1}\right)=\left\{\begin{array}{lll}
1 & \text { if } & \hat{p}\left(x_{n+1}\right) \geq 1 / 2 \\
0 & \text { if } & \hat{p}\left(x_{n+1}\right)<1 / 2
\end{array}\right.
$$

- In practice, cut points different from $1 / 2$ can be used.


## 24. The problem of binary classification. Parametric estimation

- Parametric modeling: it is assumed that $p(\boldsymbol{x})=\operatorname{Pr}(Y=1 \mid \boldsymbol{X}=\boldsymbol{x})$ is known except for a finite number of unknown parameters,

$$
p(x) \equiv p(x ; \theta), \theta \in \Theta \subseteq \mathbb{R}^{q} .
$$

- The likelihood function is

$$
L(\theta)=\prod_{i=1}^{n} \operatorname{Pr}\left(Y_{i}=y_{i} \mid \boldsymbol{X}_{i}=\boldsymbol{x}_{i}\right)=\prod_{i=1}^{n} p\left(\boldsymbol{x}_{i} ; \theta\right)^{y_{i}}\left(1-p\left(\boldsymbol{x}_{i} ; \theta\right)\right)^{1-y_{i}},
$$

with logarithm

$$
\ell(\theta)=\log L(\theta)=\sum_{i=1}^{n}\left(y_{i} \log p\left(\boldsymbol{x}_{i} ; \theta\right)+\left(1-y_{i}\right) \log \left(1-p\left(\boldsymbol{x}_{i} ; \theta\right)\right)\right) .
$$

- Let $\hat{\theta}=\arg \max _{\theta \in \Theta} \ell(\theta)$ be the maximum likelihood estimator of $\theta$.
- Then $\hat{p}(\boldsymbol{x})=p(\boldsymbol{x} ; \hat{\theta})$ is used to define the classification rule.


## 25. The problem of binary classification. Other optimization

 criteria- Maximum likelihood is not the only possibility for estimating $\theta$ in $p(x ; \theta)$.
- Alternatives:
- Minimization of the misclassification error:

$$
\hat{\theta}_{\text {Miss }}=\arg \min _{\theta \in \Theta} \sum_{i=1}^{n}\left(y_{i}-\mathbb{I}\left\{p\left(\boldsymbol{x}_{i} ; \theta\right) \geq 0.5\right\}\right)^{2} .
$$

- Least squares estimation: $\hat{\theta}_{L S}=\arg \min _{\theta \in \Theta} \sum_{i=1}^{n}\left(y_{i}-p\left(\boldsymbol{x}_{i} ; \theta\right)\right)^{2}$.
- Least absolute deviation: $\hat{\theta}_{L A D}=\arg \min _{\theta \in \Theta} \sum_{i=1}^{n}\left|y_{i}-p\left(\boldsymbol{x}_{i} ; \theta\right)\right|$.
- Penalized version of these criteria, when the statistical model $p(\boldsymbol{x} ; \theta), \theta \in \mathbb{R}^{q}$, is too flexible.


## 26. Evaluating a binary classification rule

From Wikipedia, the free encyclopedia

|  |  | Predicted condition |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: |
|  | Total population $=P+N$ | Predicted Positive (PP) | Predicted Negative (PN) | $\begin{aligned} & \text { Informedness, bookmaker } \\ & \text { informedness (BM) } \\ & =\text { TPR }+ \text { TNR }-1 \end{aligned}$ | Prevalence threshold (PT) $=\frac{\sqrt{T P R \times F P R}-F P R}{T P R-F P R}$ |
| ![](https://cdn.mathpix.com/cropped/2025_02_08_69f53f20c21a4f60a057g-29.jpg?height=132&width=22&top_left_y=384&top_left_x=104) | Positive (P) | True positive (TP), hit | False negative (FN), type II error, miss, underestimation | True positive rate (TPR), recall, sensitivity (SEN), <br> probability of detection, hit rate, power $=\frac{T P}{P}=1-F N R$ | $\begin{gathered} \text { False negative rate (FNR), } \\ \text { miss rate } \\ =\frac{F N}{P}=1-T P R \end{gathered}$ |
|  | Negative ( N ) | False positive (FP), type I error, false alarm, overestimation | True negative (TN), correct rejection | False positive rate (FPR), probability of false alarm, fall-out $=\frac{\mathrm{FP}}{\mathrm{~N}}=1-\mathrm{TNR}$ | True negative rate (TNR), specificity (SPC), selectivity $=\frac{T N}{N}=1-F P R$ |
|  | Prevalence $=\frac{P}{P+N}$ | $\begin{gathered} \text { Positive predictive value (PPV), } \\ \text { precision } \\ =\frac{T P}{P P}=1-F D R \end{gathered}$ | False omission rate $\begin{gathered} \text { (FOR) } \\ =\frac{F N}{P N}=1-N P V \end{gathered}$ | $\begin{aligned} & \text { Positive likelihood ratio (LR+) } \\ & \qquad=\frac{T P R}{F P R} \end{aligned}$ | Negative likelihood ratio (LR-) $=\frac{F N R}{T N R}$ |
|  | $\begin{gathered} \text { Accuracy (ACC) } \\ =\frac{T P+T N}{P+N} \end{gathered}$ | False discovery rate (FDR) $=\frac{F P}{P P}=1-P P V$ | Negative predictive $\begin{aligned} & \text { value }(\mathrm{NPV})=\frac{\mathrm{TN}}{\mathrm{PN}} \\ & \quad=1-\mathrm{FOR} \end{aligned}$ | $\begin{gathered} \text { Markedness }(M K) \text {, deltaP }(\Delta p) \\ =P P V+N P V-1 \end{gathered}$ | Diagnostic odds ratio (DOR) $=\frac{L R+}{L R-}$ |
|  | Balanced accuracy (BA) $=\frac{\mathrm{TPR}+\mathrm{TNR}}{2}$ | $\begin{gathered} F_{1} \text { score } \\ =\frac{2 P P V \times T P R}{P P V+T P R}=\frac{2 T P}{2 T P+F P+F N} \end{gathered}$ | $\begin{aligned} & \text { Fowlkes-Mallows } \\ & \text { index (FM) } \\ & =\sqrt{\text { PPV } \times \text { TPR }} \end{aligned}$ | Matthews correlation coefficient $\begin{aligned} & (\mathrm{MCC}) \\ & =\sqrt{\mathrm{TPR} \times \mathrm{TNR} \times \mathrm{PPV} \times N P V} \\ & -\sqrt{F N R \times F P R \times F O R \times F D R} \end{aligned}$ | Threat score (TS), critical success index (CSI), Jaccard $\text { index }=\frac{T P}{T P+F N+F P}$ |

https://en.wikipedia.org/wiki/Template:Diagnostic_testing_diagram

## 27. $k$-nn classification, in R

```
knn_class<-function(x,y,t=NULL,k=3,dist.method="euclidean"){
    nx <- length(y)
    classes <- sort(unique(y))
    nclasses <- length(classes)
    if (is.null(t)){t<-as.matrix(x)}else{t<-as.matrix(t) }
    nt <- dim(t)[1]
    Dtx <- as.matrix(dist(rbind(t,as.matrix(x)), method=dist.
method))
    Dtx <- Dtx[1:nt,nt+(1:nx)]
    hat_probs_t <- matrix(0, nrow = nt, ncol=nclasses)
    hat_y_t <- numeric(nt)
    for (i in 1:nt){
        d_t_x <- Dtx[i,]
        d_t_x_k <- sort(d_t_x,partial=k) [k]
        Ntk <- unname( which(d_t_x <= d_t_x_k) )
        for (j in 1:nclasses){
            hat_probs_t[i,j]<-sum(y[Ntk]==classes[j])/length(Ntk)
        }
        hat_y_t[i] <- classes[which.max(hat_probs_t[i,])]
    }
    return(list(hat_y_t=hat_y_t, hat_probs_t=hat_probs_t))
```


## 28. Example of $k$-nn classification

```
n <- 200; sd_eps <- . 05
x <- matrix(2*runif(2*n)-1, ncol=2)
px <- exp(-x[,1]^2-x[,2]^2)
y <- rbinom(n,size=1,prob = px)
plot(x[,1],x[,2],xlim=c(-1,1),ylim=c(-1,1), col=y+1, asp=1)
abline(h=0,v=0, col=8,lty=3)
lines(sqrt(log(2))*\operatorname{cos(seq(0, 2*pi,length=201)),}
sqrt(log(2))*sin(seq(0,2*pi,length=201)),col=8,lty=2)
k <- n/20
hat_y <- knn_class(x,y, k= k)
points(x[,1],x[,2], pch=19, cex=.5, col=hat_y$hat_y_t+1)
title(main=paste0("k-nn classification estimator, k=",k,
"\n Misclassification rate: ", mean(y!=hat_y$hat_y_t)))
```


## 29. k-nn classification estimator, $\mathbf{k = 1 0}$ <br> Misclassification rate: 0.225

![](https://cdn.mathpix.com/cropped/2025_02_08_69f53f20c21a4f60a057g-32.jpg?height=697&width=781&top_left_y=204&top_left_x=214)

## 30. Practice:

## 31. Follow the Rmd files

SimMixtNorm.Rmd and knn_class.Rmd.

Hastie, T., R. Tibshirani, and J. Friedman (2009).
The Elements of Statistical Learning (2nd ed.).
Springer.
James, G., D. Witten, T. Hastie, R. Tibshirani, and J. Taylor (2023). An Introduction to Statistical Learning with Applications in Python. Springer.


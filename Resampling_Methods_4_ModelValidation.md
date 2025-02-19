## Cross-validation and the Bootstrap

- In the section we discuss two resampling methods: cross-validation and the bootstrap.


## Cross-validation and the Bootstrap

- In the section we discuss two resampling methods: cross-validation and the bootstrap.
- These methods refit a model of interest to samples formed from the training set, in order to obtain additional information about the fitted model.


## Cross-validation and the Bootstrap

- In the section we discuss two resampling methods: cross-validation and the bootstrap.
- These methods refit a model of interest to samples formed from the training set, in order to obtain additional information about the fitted model.
- For example, they provide estimates of test-set prediction error, and the standard deviation and bias of our parameter estimates


## Training Error versus Test error

- Recall the distinction between the test error and the training error:
- The test error is the average error that results from using a statistical learning method to predict the response on a new observation, one that was not used in training the method.
- In contrast, the training error can be easily calculated by applying the statistical learning method to the observations used in its training.
- But the training error rate often is quite different from the test error rate, and in particular the former can dramatically underestimate the latter.


## Training- versus Test-Set Performance

![](https://cdn.mathpix.com/cropped/2025_02_18_d84fddb1dda73076f5eag-05.jpg?height=682&width=964&top_left_y=157&top_left_x=151)

## More on prediction-error estimates

- Best solution: a large designated test set. Often not available
- Some methods make a mathematical adjustment to the training error rate in order to estimate the test error rate. These include the $C p$ statistic, $A I C$ and BIC. They are discussed elsewhere in this course
- Here we instead consider a class of methods that estimate the test error by holding out a subset of the training observations from the fitting process, and then applying the statistical learning method to those held out observations


## Validation-set approach

- Here we randomly divide the available set of samples into two parts: a training set and a validation or hold-out set.
- The model is fit on the training set, and the fitted model is used to predict the responses for the observations in the validation set.
- The resulting validation-set error provides an estimate of the test error. This is typically assessed using MSE in the case of a quantitative response and misclassification rate in the case of a qualitative (discrete) response.


## The Validation process

![](https://cdn.mathpix.com/cropped/2025_02_18_d84fddb1dda73076f5eag-08.jpg?height=187&width=830&top_left_y=310&top_left_x=205)

A random splitting into two halves: left part is training set, right part is validation set

## Example: automobile data

- Want to compare linear vs higher-order polynomial terms in a linear regression
- We randomly split the 392 observations into two sets, a training set containing 196 of the data points, and a validation set containing the remaining 196 observations.
![](https://cdn.mathpix.com/cropped/2025_02_18_d84fddb1dda73076f5eag-09.jpg?height=379&width=954&top_left_y=466&top_left_x=140)

Left panel shows single split; right panel shows multiple splits

## Drawbacks of validation set approach

- the validation estimate of the test error can be highly variable, depending on precisely which observations are included in the training set and which observations are included in the validation set.
- In the validation approach, only a subset of the observations - those that are included in the training set rather than in the validation set - are used to fit the model.
- This suggests that the validation set error may tend to overestimate the test error for the model fit on the entire data set.


## Drawbacks of validation set approach

- the validation estimate of the test error can be highly variable, depending on precisely which observations are included in the training set and which observations are included in the validation set.
- In the validation approach, only a subset of the observations - those that are included in the training set rather than in the validation set - are used to fit the model.
- This suggests that the validation set error may tend to overestimate the test error for the model fit on the entire data set. Why?


## $K$-fold Cross-validation

- Widely used approach for estimating test error.
- Estimates can be used to select best model, and to give an idea of the test error of the final chosen model.
- Idea is to randomly divide the data into $K$ equal-sized parts. We leave out part $k$, fit the model to the other $K-1$ parts (combined), and then obtain predictions for the left-out $k$ th part.
- This is done in turn for each part $k=1,2, \ldots K$, and then the results are combined.


## $K$-fold Cross-validation in detail

Divide data into $K$ roughly equal-sized parts ( $K=5$ here)
![](https://cdn.mathpix.com/cropped/2025_02_18_d84fddb1dda73076f5eag-13.jpg?height=230&width=871&top_left_y=396&top_left_x=201)

## The details

- Let the $K$ parts be $C_{1}, C_{2}, \ldots C_{K}$, where $C_{k}$ denotes the indices of the observations in part $k$. There are $n_{k}$ observations in part $k$ : if $N$ is a multiple of $K$, then $n_{k}=n / K$.
- Compute

$$
\mathrm{CV}_{(K)}=\sum_{k=1}^{K} \frac{n_{k}}{n} \mathrm{MSE}_{k}
$$

where $\mathrm{MSE}_{k}=\sum_{i \in C_{k}}\left(y_{i}-\hat{y}_{i}\right)^{2} / n_{k}$, and $\hat{y}_{i}$ is the fit for observation $i$, obtained from the data with part $k$ removed.

## The details

- Let the $K$ parts be $C_{1}, C_{2}, \ldots C_{K}$, where $C_{k}$ denotes the indices of the observations in part $k$. There are $n_{k}$ observations in part $k$ : if $N$ is a multiple of $K$, then $n_{k}=n / K$.
- Compute

$$
\mathrm{CV}_{(K)}=\sum_{k=1}^{K} \frac{n_{k}}{n} \mathrm{MSE}_{k}
$$

where $\mathrm{MSE}_{k}=\sum_{i \in C_{k}}\left(y_{i}-\hat{y}_{i}\right)^{2} / n_{k}$, and $\hat{y}_{i}$ is the fit for observation $i$, obtained from the data with part $k$ removed.

- Setting $K=n$ yields $n$-fold or leave-one out cross-validation (LOOCV).


## A nice special case!

- With least-squares linear or polynomial regression, an amazing shortcut makes the cost of LOOCV the same as that of a single model fit! The following formula holds:

$$
\mathrm{CV}_{(n)}=\frac{1}{n} \sum_{i=1}^{n}\left(\frac{y_{i}-\hat{y}_{i}}{1-h_{i}}\right)^{2}
$$

where $\hat{y}_{i}$ is the $i$ th fitted value from the original least squares fit, and $h_{i}$ is the leverage (diagonal of the "hat" matrix; see book for details.) This is like the ordinary MSE, except the $i$ th residual is divided by $1-h_{i}$.

## A nice special case!

- With least-squares linear or polynomial regression, an amazing shortcut makes the cost of LOOCV the same as that of a single model fit! The following formula holds:

$$
\mathrm{CV}_{(n)}=\frac{1}{n} \sum_{i=1}^{n}\left(\frac{y_{i}-\hat{y}_{i}}{1-h_{i}}\right)^{2}
$$

where $\hat{y}_{i}$ is the $i$ th fitted value from the original least squares fit, and $h_{i}$ is the leverage (diagonal of the "hat" matrix; see book for details.) This is like the ordinary MSE, except the $i$ th residual is divided by $1-h_{i}$.

- LOOCV sometimes useful, but typically doesn't shake up the data enough. The estimates from each fold are highly correlated and hence their average can have high variance.
- a better choice is $K=5$ or 10 .


## Auto data revisited

![](https://cdn.mathpix.com/cropped/2025_02_18_d84fddb1dda73076f5eag-18.jpg?height=445&width=509&top_left_y=263&top_left_x=94)

10-fold CV
![](https://cdn.mathpix.com/cropped/2025_02_18_d84fddb1dda73076f5eag-18.jpg?height=395&width=509&top_left_y=310&top_left_x=623)

## True and estimated test MSE for the simulated data

![](https://cdn.mathpix.com/cropped/2025_02_18_d84fddb1dda73076f5eag-19.jpg?height=419&width=348&top_left_y=292&top_left_x=90)
![](https://cdn.mathpix.com/cropped/2025_02_18_d84fddb1dda73076f5eag-19.jpg?height=415&width=339&top_left_y=294&top_left_x=446)
![](https://cdn.mathpix.com/cropped/2025_02_18_d84fddb1dda73076f5eag-19.jpg?height=414&width=335&top_left_y=293&top_left_x=803)

## Other issues with Cross-validation

- Since each training set is only $(K-1) / K$ as big as the original training set, the estimates of prediction error will typically be biased upward.


## Other issues with Cross-validation

- Since each training set is only $(K-1) / K$ as big as the original training set, the estimates of prediction error will typically be biased upward. Why?


## Other issues with Cross-validation

- Since each training set is only $(K-1) / K$ as big as the original training set, the estimates of prediction error will typically be biased upward. Why?
- This bias is minimized when $K=n$ (LOOCV), but this estimate has high variance, as noted earlier.
- $K=5$ or 10 provides a good compromise for this bias-variance tradeoff.


## Cross-Validation for Classification Problems

- We divide the data into $K$ roughly equal-sized parts $C_{1}, C_{2}, \ldots C_{K} . C_{k}$ denotes the indices of the observations in part $k$. There are $n_{k}$ observations in part $k$ : if $n$ is a multiple of $K$, then $n_{k}=n / K$.
- Compute

$$
\mathrm{CV}_{K}=\sum_{k=1}^{K} \frac{n_{k}}{n} \operatorname{Err}_{k}
$$

where $\operatorname{Err}_{k}=\sum_{i \in C_{k}} I\left(y_{i} \neq \hat{y}_{i}\right) / n_{k}$.

- The estimated standard deviation of $\mathrm{CV}_{K}$ is

$$
\widehat{\mathrm{SE}}\left(\mathrm{CV}_{K}\right)=\sqrt{\frac{1}{K} \sum_{k=1}^{K} \frac{\left(\operatorname{Err}_{k}-\overline{\operatorname{Err}_{k}}\right)^{2}}{K-1}}
$$

- This is a useful estimate, but strictly speaking, not quite valid.


## Cross-Validation for Classification Problems

- We divide the data into $K$ roughly equal-sized parts $C_{1}, C_{2}, \ldots C_{K} . C_{k}$ denotes the indices of the observations in part $k$. There are $n_{k}$ observations in part $k$ : if $n$ is a multiple of $K$, then $n_{k}=n / K$.
- Compute

$$
\mathrm{CV}_{K}=\sum_{k=1}^{K} \frac{n_{k}}{n} \operatorname{Err}_{k}
$$

where $\operatorname{Err}_{k}=\sum_{i \in C_{k}} I\left(y_{i} \neq \hat{y}_{i}\right) / n_{k}$.

- The estimated standard deviation of $\mathrm{CV}_{K}$ is

$$
\widehat{\mathrm{SE}}\left(\mathrm{CV}_{K}\right)=\sqrt{\frac{1}{K} \sum_{k=1}^{K} \frac{\left(\operatorname{Err}_{k}-\overline{\operatorname{Err}_{k}}\right)^{2}}{K-1}}
$$

- This is a useful estimate, but strictly speaking, not quite valid. Why not?


## Cross-validation: right and wrong

- Consider a simple classifier applied to some two-class data:

1. Starting with 5000 predictors and 50 samples, find the 100 predictors having the largest correlation with the class labels.
2. We then apply a classifier such as logistic regression, using only these 100 predictors.

How do we estimate the test set performance of this classifier?

## Cross-validation: right and wrong

- Consider a simple classifier applied to some two-class data:

1. Starting with 5000 predictors and 50 samples, find the 100 predictors having the largest correlation with the class labels.
2. We then apply a classifier such as logistic regression, using only these 100 predictors.

How do we estimate the test set performance of this classifier?

Can we apply cross-validation in step 2, forgetting about step 1?

## NO!

- This would ignore the fact that in Step 1, the procedure has already seen the labels of the training data, and made use of them. This is a form of training and must be included in the validation process.
- It is easy to simulate realistic data with the class labels independent of the outcome, so that true test error $=50 \%$, but the CV error estimate that ignores Step 1 is zero!


## NO!

- This would ignore the fact that in Step 1, the procedure has already seen the labels of the training data, and made use of them. This is a form of training and must be included in the validation process.
- It is easy to simulate realistic data with the class labels independent of the outcome, so that true test error $=50 \%$, but the CV error estimate that ignores Step 1 is zero! Try to do this yourself


## NO!

- This would ignore the fact that in Step 1, the procedure has already seen the labels of the training data, and made use of them. This is a form of training and must be included in the validation process.
- It is easy to simulate realistic data with the class labels independent of the outcome, so that true test error $=50 \%$, but the CV error estimate that ignores Step 1 is zero! Try to do this yourself
- We have seen this error made in many high profile genomics papers.


## The Wrong and Right Way

- Wrong: Apply cross-validation in step 2.
- Right: Apply cross-validation to steps 1 and 2.


## Wrong Way

![](https://cdn.mathpix.com/cropped/2025_02_18_d84fddb1dda73076f5eag-31.jpg?height=503&width=1068&top_left_y=229&top_left_x=101)

## Right Way

![](https://cdn.mathpix.com/cropped/2025_02_18_d84fddb1dda73076f5eag-32.jpg?height=499&width=1036&top_left_y=234&top_left_x=112)

## The Bootstrap

- The bootstrap is a flexible and powerful statistical tool that can be used to quantify the uncertainty associated with a given estimator or statistical learning method.
- For example, it can provide an estimate of the standard error of a coefficient, or a confidence interval for that coefficient.


## Where does the name came from?

- The use of the term bootstrap derives from the phrase to pull oneself up by one's bootstraps, widely thought to be based on one of the eighteenth century "The Surprising Adventures of Baron Munchausen" by Rudolph Erich Raspe:

The Baron had fallen to the bottom of a deep lake. Just when it looked like all was lost, he thought to pick himself up by his own bootstraps.

- It is not the same as the term "bootstrap" used in computer science meaning to "boot" a computer from a set of core instructions, though the derivation is similar.


## A simple example

- Suppose that we wish to invest a fixed sum of money in two financial assets that yield returns of $X$ and $Y$, respectively, where $X$ and $Y$ are random quantities.
- We will invest a fraction $\alpha$ of our money in $X$, and will invest the remaining $1-\alpha$ in $Y$.
- We wish to choose $\alpha$ to minimize the total risk, or variance, of our investment. In other words, we want to minimize $\operatorname{Var}(\alpha X+(1-\alpha) Y)$.


## A simple example

- Suppose that we wish to invest a fixed sum of money in two financial assets that yield returns of $X$ and $Y$, respectively, where $X$ and $Y$ are random quantities.
- We will invest a fraction $\alpha$ of our money in $X$, and will invest the remaining $1-\alpha$ in $Y$.
- We wish to choose $\alpha$ to minimize the total risk, or variance, of our investment. In other words, we want to minimize $\operatorname{Var}(\alpha X+(1-\alpha) Y)$.
- One can show that the value that minimizes the risk is given by

$$
\alpha=\frac{\sigma_{Y}^{2}-\sigma_{X Y}}{\sigma_{X}^{2}+\sigma_{Y}^{2}-2 \sigma_{X Y}}
$$

where $\sigma_{X}^{2}=\operatorname{Var}(X), \sigma_{Y}^{2}=\operatorname{Var}(Y)$, and $\sigma_{X Y}=\operatorname{Cov}(X, Y)$.

## Example continued

- But the values of $\sigma_{X}^{2}, \sigma_{Y}^{2}$, and $\sigma_{X Y}$ are unknown.
- We can compute estimates for these quantities, $\hat{\sigma}_{X}^{2}, \hat{\sigma}_{Y}^{2}$, and $\hat{\sigma}_{X Y}$, using a data set that contains measurements for $X$ and $Y$.
- We can then estimate the value of $\alpha$ that minimizes the variance of our investment using

$$
\hat{\alpha}=\frac{\hat{\sigma}_{Y}^{2}-\hat{\sigma}_{X Y}}{\hat{\sigma}_{X}^{2}+\hat{\sigma}_{Y}^{2}-2 \hat{\sigma}_{X Y}}
$$

## Example continued

![](https://cdn.mathpix.com/cropped/2025_02_18_d84fddb1dda73076f5eag-38.jpg?height=595&width=635&top_left_y=159&top_left_x=306)

Each panel displays 100 simulated returns for investments $X$ and $Y$. From left to right and top to bottom, the resulting estimates for $\alpha$ are 0.576, 0.532, 0.657, and 0.651.

## Example continued

- To estimate the standard deviation of $\hat{\alpha}$, we repeated the process of simulating 100 paired observations of $X$ and $Y$, and estimating $\alpha 1,000$ times.
- We thereby obtained 1,000 estimates for $\alpha$, which we can call $\hat{\alpha}_{1}, \hat{\alpha}_{2}, \ldots, \hat{\alpha}_{1000}$.
- The left-hand panel of the Figure on slide 29 displays a histogram of the resulting estimates.
- For these simulations the parameters were set to $\sigma_{X}^{2}=1, \sigma_{Y}^{2}=1.25$, and $\sigma_{X Y}=0.5$, and so we know that the true value of $\alpha$ is 0.6 (indicated by the red line).


## Example continued

- The mean over all 1,000 estimates for $\alpha$ is

$$
\bar{\alpha}=\frac{1}{1000} \sum_{r=1}^{1000} \hat{\alpha}_{r}=0.5996,
$$

very close to $\alpha=0.6$, and the standard deviation of the estimates is

$$
\sqrt{\frac{1}{1000-1} \sum_{r=1}^{1000}\left(\hat{\alpha}_{r}-\bar{\alpha}\right)^{2}}=0.083
$$

- This gives us a very good idea of the accuracy of $\hat{\alpha}$ : $\mathrm{SE}(\hat{\alpha}) \approx 0.083$.
- So roughly speaking, for a random sample from the population, we would expect $\hat{\alpha}$ to differ from $\alpha$ by approximately 0.08 , on average.


## Results

![](https://cdn.mathpix.com/cropped/2025_02_18_d84fddb1dda73076f5eag-41.jpg?height=343&width=247&top_left_y=208&top_left_x=217)
![](https://cdn.mathpix.com/cropped/2025_02_18_d84fddb1dda73076f5eag-41.jpg?height=336&width=248&top_left_y=213&top_left_x=503)
![](https://cdn.mathpix.com/cropped/2025_02_18_d84fddb1dda73076f5eag-41.jpg?height=313&width=274&top_left_y=216&top_left_x=760)

Left: A histogram of the estimates of $\alpha$ obtained by generating 1,000 simulated data sets from the true population. Center: A histogram of the estimates of $\alpha$ obtained from 1,000 bootstrap samples from a single data set. Right: The estimates of $\alpha$ displayed in the left and center panels are shown as boxplots. In each panel, the pink line indicates the true value of $\alpha$.

## Now back to the real world

- The procedure outlined above cannot be applied, because for real data we cannot generate new samples from the original population.
- However, the bootstrap approach allows us to use a computer to mimic the process of obtaining new data sets, so that we can estimate the variability of our estimate without generating additional samples.
- Rather than repeatedly obtaining independent data sets from the population, we instead obtain distinct data sets by repeatedly sampling observations from the original data set with replacement.
- Each of these "bootstrap data sets" is created by sampling with replacement, and is the same size as our original dataset. As a result some observations may appear more than once in a given bootstrap data set and some not at all.


## Example with just 3 observations

![](https://cdn.mathpix.com/cropped/2025_02_18_d84fddb1dda73076f5eag-43.jpg?height=456&width=504&top_left_y=161&top_left_x=306)

A graphical illustration of the bootstrap approach on a small sample containing $n=3$ observations. Each bootstrap data set contains $n$ observations, sampled with replacement from the original data set. Each bootstrap data set is used to obtain an estimate of $\alpha$

- Denoting the first bootstrap data set by $Z^{* 1}$, we use $Z^{* 1}$ to produce a new bootstrap estimate for $\alpha$, which we call $\hat{\alpha}^{* 1}$
- This procedure is repeated $B$ times for some large value of $B$ (say 100 or 1000), in order to produce $B$ different bootstrap data sets, $Z^{* 1}, Z^{* 2}, \ldots, Z^{* B}$, and $B$ corresponding $\alpha$ estimates, $\hat{\alpha}^{* 1}, \hat{\alpha}^{* 2}, \ldots, \hat{\alpha}^{* B}$.
- We estimate the standard error of these bootstrap estimates using the formula

$$
\mathrm{SE}_{B}(\hat{\alpha})=\sqrt{\frac{1}{B-1} \sum_{r=1}^{B}\left(\hat{\alpha}^{* r}-\overline{\hat{\alpha}}^{*}\right)^{2}}
$$

- This serves as an estimate of the standard error of $\hat{\alpha}$ estimated from the original data set. See center and right panels of Figure on slide 29. Bootstrap results are in blue. For this example $\mathrm{SE}_{B}(\hat{\alpha})=0.087$.


## A general picture for the bootstrap

![](https://cdn.mathpix.com/cropped/2025_02_18_d84fddb1dda73076f5eag-45.jpg?height=456&width=1072&top_left_y=248&top_left_x=98)

## The bootstrap in general

- In more complex data situations, figuring out the appropriate way to generate bootstrap samples can require some thought.
- For example, if the data is a time series, we can't simply sample the observations with replacement (why not?).


## The bootstrap in general

- In more complex data situations, figuring out the appropriate way to generate bootstrap samples can require some thought.
- For example, if the data is a time series, we can't simply sample the observations with replacement (why not?).
- We can instead create blocks of consecutive observations, and sample those with replacements. Then we paste together sampled blocks to obtain a bootstrap dataset.


## Other uses of the bootstrap

- Primarily used to obtain standard errors of an estimate.
- Also provides approximate confidence intervals for a population parameter. For example, looking at the histogram in the middle panel of the Figure on slide 29, the $5 \%$ and $95 \%$ quantiles of the 1000 values is (.43, .72 ).
- This represents an approximate $90 \%$ confidence interval for the true $\alpha$.


## Other uses of the bootstrap

- Primarily used to obtain standard errors of an estimate.
- Also provides approximate confidence intervals for a population parameter. For example, looking at the histogram in the middle panel of the Figure on slide 29, the $5 \%$ and $95 \%$ quantiles of the 1000 values is (.43, .72 ).
- This represents an approximate $90 \%$ confidence interval for the true $\alpha$. How do we interpret this confidence interval?


## Other uses of the bootstrap

- Primarily used to obtain standard errors of an estimate.
- Also provides approximate confidence intervals for a population parameter. For example, looking at the histogram in the middle panel of the Figure on slide 29, the $5 \%$ and $95 \%$ quantiles of the 1000 values is (.43, .72 ).
- This represents an approximate $90 \%$ confidence interval for the true $\alpha$. How do we interpret this confidence interval?
- The above interval is called a Bootstrap Percentile confidence interval. It is the simplest method (among many approaches) for obtaining a confidence interval from the bootstrap.


## Can the bootstrap estimate prediction error?

- In cross-validation, each of the $K$ validation folds is distinct from the other $K-1$ folds used for training: there is no overlap. This is crucial for its success.


## Can the bootstrap estimate prediction error?

- In cross-validation, each of the $K$ validation folds is distinct from the other $K-1$ folds used for training: there is no overlap. This is crucial for its success. Why?


## Can the bootstrap estimate prediction error?

- In cross-validation, each of the $K$ validation folds is distinct from the other $K-1$ folds used for training: there is no overlap. This is crucial for its success. Why?
- To estimate prediction error using the bootstrap, we could think about using each bootstrap dataset as our training sample, and the original sample as our validation sample.
- But each bootstrap sample has significant overlap with the original data. About two-thirds of the original data points appear in each bootstrap sample.


## Can the bootstrap estimate prediction error?

- In cross-validation, each of the $K$ validation folds is distinct from the other $K-1$ folds used for training: there is no overlap. This is crucial for its success. Why?
- To estimate prediction error using the bootstrap, we could think about using each bootstrap dataset as our training sample, and the original sample as our validation sample.
- But each bootstrap sample has significant overlap with the original data. About two-thirds of the original data points appear in each bootstrap sample. Can you prove this?


## Can the bootstrap estimate prediction error?

- In cross-validation, each of the $K$ validation folds is distinct from the other $K-1$ folds used for training: there is no overlap. This is crucial for its success. Why?
- To estimate prediction error using the bootstrap, we could think about using each bootstrap dataset as our training sample, and the original sample as our validation sample.
- But each bootstrap sample has significant overlap with the original data. About two-thirds of the original data points appear in each bootstrap sample. Can you prove this?
- This will cause the bootstrap to seriously underestimate the true prediction error.


## Can the bootstrap estimate prediction error?

- In cross-validation, each of the $K$ validation folds is distinct from the other $K-1$ folds used for training: there is no overlap. This is crucial for its success. Why?
- To estimate prediction error using the bootstrap, we could think about using each bootstrap dataset as our training sample, and the original sample as our validation sample.
- But each bootstrap sample has significant overlap with the original data. About two-thirds of the original data points appear in each bootstrap sample. Can you prove this?
- This will cause the bootstrap to seriously underestimate the true prediction error. Why?


## Can the bootstrap estimate prediction error?

- In cross-validation, each of the $K$ validation folds is distinct from the other $K-1$ folds used for training: there is no overlap. This is crucial for its success. Why?
- To estimate prediction error using the bootstrap, we could think about using each bootstrap dataset as our training sample, and the original sample as our validation sample.
- But each bootstrap sample has significant overlap with the original data. About two-thirds of the original data points appear in each bootstrap sample. Can you prove this?
- This will cause the bootstrap to seriously underestimate the true prediction error. Why?
- The other way around- with original sample = training sample, bootstrap dataset $=$ validation sample - is worse!


## Removing the overlap

- Can partly fix this problem by only using predictions for those observations that did not (by chance) occur in the current bootstrap sample.
- But the method gets complicated, and in the end, cross-validation provides a simpler, more attractive approach for estimating prediction error.


## Pre-validation

- In microarray and other genomic studies, an important problem is to compare a predictor of disease outcome derived from a large number of "biomarkers" to standard clinical predictors.
- Comparing them on the same dataset that was used to derive the biomarker predictor can lead to results strongly biased in favor of the biomarker predictor.


## Pre-validation

- In microarray and other genomic studies, an important problem is to compare a predictor of disease outcome derived from a large number of "biomarkers" to standard clinical predictors.
- Comparing them on the same dataset that was used to derive the biomarker predictor can lead to results strongly biased in favor of the biomarker predictor.
- Pre-validation can be used to make a fairer comparison between the two sets of predictors.


## Motivating example

An example of this problem arose in the paper of van't Veer et al. Nature (2002). Their microarray data has 4918 genes measured over 78 cases, taken from a study of breast cancer. There are 44 cases in the good prognosis group and 34 in the poor prognosis group. A "microarray" predictor was constructed as follows:

1. 70 genes were selected, having largest absolute correlation with the 78 class labels.
2. Using these 70 genes, a nearest-centroid classifier $C(x)$ was constructed.
3. Applying the classifier to the 78 microarrays gave a dichotomous predictor $z_{i}=C\left(x_{i}\right)$ for each case $i$.

## Results

Comparison of the microarray predictor with some clinical predictors, using logistic regression with outcome prognosis:

| Model | Coef | Stand. Err. | Z score | p-value |
| :--- | ---: | ---: | ---: | ---: |
| Re-use |  |  |  |  |
| microarray | 4.096 | 1.092 | 3.753 | 0.000 |
| angio | 1.208 | 0.816 | 1.482 | 0.069 |
| er | -0.554 | 1.044 | -0.530 | 0.298 |
| grade | -0.697 | 1.003 | -0.695 | 0.243 |
| pr | 1.214 | 1.057 | 1.149 | 0.125 |
| age | -1.593 | 0.911 | -1.748 | 0.040 |
| size | 1.483 | 0.732 | 2.026 | 0.021 |
|  | Pre-validated |  |  |  |
|  |  |  |  |  |
| microarray | 1.549 | 0.675 | 2.296 | 0.011 |
| angio | 1.589 | 0.682 | 2.329 | 0.010 |
| er | -0.617 | 0.894 | -0.690 | 0.245 |
| grade | 0.719 | 0.720 | 0.999 | 0.159 |
| pr | 0.537 | 0.863 | 0.622 | 0.267 |
| age | -1.471 | 0.701 | -2.099 | 0.018 |
| size | 0.998 | 0.594 | 1.681 | 0.046 |

## Idea behind Pre-validation

- Designed for comparison of adaptively derived predictors to fixed, pre-defined predictors.
- The idea is to form a "pre-validated" version of the adaptive predictor: specifically, a "fairer" version that hasn't "seen" the response $y$.


## Pre-validation process

![](https://cdn.mathpix.com/cropped/2025_02_18_d84fddb1dda73076f5eag-64.jpg?height=760&width=812&top_left_y=139&top_left_x=229)

## Pre-validation in detail for this example

1. Divide the cases up into $K=13$ equal-sized parts of 6 cases each.
2. Set aside one of parts. Using only the data from the other 12 parts, select the features having absolute correlation at least .3 with the class labels, and form a nearest centroid classification rule.
3. Use the rule to predict the class labels for the 13th part
4. Do steps 2 and 3 for each of the 13 parts, yielding a "pre-validated" microarray predictor $\tilde{z}_{i}$ for each of the 78 cases.
5. Fit a logistic regression model to the pre-validated microarray predictor and the 6 clinical predictors.

## The Bootstrap versus Permutation tests

- The bootstrap samples from the estimated population, and uses the results to estimate standard errors and confidence intervals.
- Permutation methods sample from an estimated null distribution for the data, and use this to estimate p-values and False Discovery Rates for hypothesis tests.
- The bootstrap can be used to test a null hypothesis in simple situations. Eg if $\theta=0$ is the null hypothesis, we check whether the confidence interval for $\theta$ contains zero.
- Can also adapt the bootstrap to sample from a null distribution (See Efron and Tibshirani book "An Introduction to the Bootstrap" (1993), chapter 16) but there's no real advantage over permutations.


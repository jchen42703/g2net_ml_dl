# Rocket Time Series Classifier

- [Rocket Classifier Paper](https://link.springer.com/article/10.1007/s10618-020-00701-z)
- [Official repo](https://github.com/angus924/rocket)
- [`sktime` implementation of Rocket](https://www.sktime.org/en/stable/api_reference/auto_generated/sktime.classification.kernel_based.ROCKETClassifier.html#sktime.classification.kernel_based.ROCKETClassifier)
- Uses random convolutional kernels to extract features --> classifies

## Random Convolutional Kernels

- It may be that learning 'good' kernels is difficult on small datasets. Random convolutional kernels may have an advantage in this context (see Jarrett et al. 2009; Yosinski et al. 2014).
- The idea of using convolutional kernels as a transform, and using the transformed features as the input to another classifier is well established (see, e.g., Bengio et al. 2013, p. 1803).
- **Example Prev Approach: Franceschi et al.**

  - Unsupervised learning
  - conv kernels as preprocessing for time series input
  - SVM to process output and generates classification

- **Pattern:** Use convolutional kernels as preprocessing transform --> pass as input to a linear classifier

> However, there are key differences between Rocket and other methods using random convolutional kernels in terms of: (a) the configuration of the convolutional kernels (bias, length, dilation, and padding); (b) the use of nonlinearities; (c) pooling; and (d) the need with other methods to ‘tune’ certain hyperparameters.

## Shapelets and random shapelets

- Shapelets are similar to conv kernels because both **discriminate between classes based on the similarity of input time series to a set of patterns**
- shapelets are typically **sampled from the input**, and are typically longer subsequences (up to the length of the input)
- The convolutional kernels used in Rocket are short, and are **not sampled from the input.**

## Method

- Rocket transforms time series using a large number of random convolutional kernels, i.e., kernels with random length, weights, bias, dilation, and padding.
- The transformed features are used to train a linear classifier.
- all but the largest datasets --> use a **ridge regression classifier**
- large datasets --> use **logistic regression**
  - Note: Data should be stationary when using a logistic regression (- Joseph :D)

## How Rocket differentiates itself

- Rocket uses random kernel length, dilation, and padding.
  - kernel dilation, in particular, is of critical importance to the high accuracy achieved by Rocket
- Rocket does not use any nonlinearities, and uses both global max pooling as well as a very different form of pooling, that is, the proportion of positive values or ppv
  - use of ppv has the single largest effect on accuracy of any aspect of Rocket
- also uses bias differently
  - bias acts as a kind of ‘threshold’ for ppv
- **only hyperparameter for Rocket is the number of convolutional kernels, k**
  - As seen in [sktime RocketClassifier](https://www.sktime.org/en/stable/api_reference/auto_generated/sktime.classification.kernel_based.ROCKETClassifier.html#sktime.classification.kernel_based.ROCKETClassifier)
- **Summary of how Rocket is different from CNNs or other kernel approaches**
  - **Rocket uses a very large number of kernels.** As there is only a single layer of kernels, and as the kernel weights are **not learned**, the cost of computing the convolutions is low, and it is possible to use a very large number of kernels with relatively little computational expense.
  - **Rocket uses a massive variety of kernels.** In contrast to typical convolutional neural networks, where it is common for groups of kernels to share the same size, dilation, and padding, for Rocket each kernel has random length, dilation, and padding, as well as random weights and bias.
  - **In particular, Rocket makes key use of kernel dilation.** Dilation is sampled randomly instead of increasing exponentially with depth.
  - **Global max pooling (?) + ppv**: the proportion of positive values (or ppv)
    - enables a classifier to weight the prevalence of a given pattern within a time series.
    - `ppv` is what gives Rocket its great performance
  - **no hidden layers, or any nonlinearities**
  - **no connections between convolutional kernels:** features produced by Rocket are independent of each other
  - no specific mandated classifier (can be flexible with classifier)

**For the experiments studying scalability, we integrate Rocket with logistic regression and Adam, implemented using PyTorch**

## Limitations

- However, Rocket is at least usable for very large datasets, whereas many existing methods for time series classification are not, and we note that accuracy on the Satellite Image Time Series dataset only seems to plateau after approximately half a million training examples (see Sect. 4.2). **Rocket is currently only configured to work with univariate time series.**

## Specific for Procedure

- **Kernel sampling:**

  - length is sampled randomly from `[7, 9, 11]`
  - weights are sampled from normal distribution
  - bias is sampled from a uniform distribution `U(-1, 1)`
  - dilation is sampled from an exponential scale: `d = floor(2^x)`
  - padding: sample whether or not to pad (50/50)
    - if pad, then zero pad until the middle element of the kernel is centered on every point in the time series
  - stride is always 1

- use Rocket in conjunction with logistic regression
- transform is performed in tranches (?), further divided into minibatches for training.
- **Each time series is normalised to have a zero mean and unit standard deviation.**
- We shuffle the original training set once, and **draw increasingly-large subsets from the shuffled training data.**
  - train the model for at least one epoch for each subset size
  - stop training (after the first epoch) if validation loss has failed to improve after 20 updates.
- In practice, while training may continue for 40 or 50 epochs for smaller subset sizes, training converges within a single pass for anything more than approximately 16,000 training examples.
- Optimisation is performed using Adam
- perform a minimal search on the initial learning rate to ensure that training loss does not diverge.
  - LR schedule: The learning rate is halved if training loss fails to improve after 100 updates (only relevant for larger subset sizes).

## Results

- Much faster training time than alternatives
  - What about inference?
- Proximity Forest and TS-CHIEF more scalable than HIVE-COTE
  - ROCKET still gets similar performance to them

# MiniRocket

## Relevant Resources

- https://towardsdatascience.com/minirocket-fast-er-and-accurate-time-series-classification-cdacca2dcbfa

## MiniRocket v. Rocket

- Rocket 2 hours to run on datasets --> MiniRocket took only 8 minutes
- Don't need to preprocess with MiniRocket

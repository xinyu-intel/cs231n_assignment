---
output:
  pdf_document: default
  html_document: default
---
Build Multi-layer Neural Network from Scratch
================

## Background

Computer Vision has become ubiquitous in our society, with applications in search, image understanding, apps, mapping, medicine, drones, and self-driving cars. Core to many of these applications are visual recognition tasks such as image classification, localization and detection. Recent developments in neural network (aka “deep learning”) approaches have greatly advanced the performance of these state-of-the-art visual recognition systems. 

The biggest advantage of Deep Neural Network is to extract and learn features automatically by deep layers architecture, especially for these complex and high-dimensional data that feature engineers can’t capture easily, examples in [Kaggle](http://blog.kaggle.com/2014/08/01/learning-from-the-best/). Therefore, DNN is also very attractive to data scientists and there are lots of successful cases as well in classification, time series, and recommendation system. In CRAN and R’s community, there are several popular and mature DNN packages including [nnet](https://cran.r-project.org/web/packages/nnet/index.html), [nerualnet](https://cran.r-project.org/web/packages/neuralnet/), [H2O](https://cran.r-project.org/web/packages/h2o/index.html), [DARCH](https://cran.r-project.org/web/packages/darch/index.html), [deepnet](https://cran.r-project.org/web/packages/deepnet/index.html) and [mxnet](https://github.com/dmlc/mxnet),  and I strong recommend [H2O DNN algorithm and R interface](http://www.h2o.ai/verticals/algos/deep-learning/).

In this post, we will focus on multi-layer neural networks. Some inspiration comes from the excellent post ["R for Deep Learning (I): Build Fully Connected Neural Network from Scratch"](http://www.parallelr.com/r-deep-neural-network-from-scratch/) on [ParallelR](http://www.parallelr.com/blog/), which induced a very simple and typical neural network with 1 input layer, 2 hidden layers, and 1 output layer. My work is to try to extend single hidden layer network to multi-hidden layers and use different activation functions.

**So, why we need to build DNN from scratch at all?**

– Understand how neural network works

Using existing DNN package, you only need one line R code for your DNN model in most of the time and there is [an example](http://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/) by neuralnet. For the inexperienced user, however, the processing and results may be difficult to understand.  Therefore, it will be a valuable practice to implement your own network in order to understand more details from mechanism and computation views.

– Build specified network with your new ideas

DNN is one of rapidly developing area. Lots of novel works and research results are published in the top journals and Internet every week, and the users also have their specified neural network configuration to meet their problems such as different activation functions, loss functions, regularization, and connected graph. On the other hand, the existing packages are definitely behind the latest researches, and almost all existing packages are written in C/C++, Java so it’s not flexible to apply latest changes and your ideas into the packages.

– Debug and visualize network and data

As we mentioned, the existing DNN package is highly assembled and written by low-level languages so that it’s a nightmare to debug the network layer by layer or node by node. Even it’s not easy to visualize the results in each layer, monitor the data or weights changes during training, and show the discovered patterns in the network.

## Fundamental Concepts and Components

A very simple and typical neural network is shown below with 1 input layer, 2 hidden layers, and 1 output layer. Mostly, when researchers talk about network’s architecture, it refers to the configuration of DNN, such as how many layers in the network, how many neurons in each layer, what kind of activation, loss function, and regularization are used.

![dnn_architecture](http://www.parallelr.com/wp-content/uploads/2016/02/dnn_architecture.png)

Since this post mainly uses sigmoid function as activation function, so we can thought neural network as a series of logistic regressions stacked on top of each other. Hidden-layer let a neural-network generate non-linearities, which states that a network with just one hidden layer can approximate any linear or non-linear function. The number of hidden-layers can go into the hundreds.

The hidden-layer also means that our loss function is not convex in parameters and we can't roll down a smooth-hill to get to the bottom. Instead of using Gradient Descent we will use Stochastic Gradient Descent (SGD), which basically shuffles the observations (random/stochastic) and updates the gradient after each mini-batch (generally much less than total number of observations) has been propagated through the network. 

Now, we can now create a multi-neural-network from scratch in R using four functions.

**Weights and Bias**

In our R implementation, we represent weights and bias by the matrix. Weight size is defined by,

  (number of neurons layer M) X (number of neurons in layer M+1)
  
and weights are initialized by random number from rnorm. Bias is just a one dimension matrix with the same size of  neurons and set to zero. Other initialization approaches, such as calibrating the variances with 1/sqrt(n) and sparse initialization, are introduced in [weight initialization](http://cs231n.github.io/neural-networks-2/#init) part of Stanford CS231n.

```{r}
# In R, we can store these parameters in lists.
biases <- lapply(seq_along(listb), function(idx){
  r <- listb[[idx]]
  matrix(0, nrow=r, ncol=1)
  })

weights <- lapply(seq_along(listb), function(idx){
  c <- listw[[idx]]
  r <- listb[[idx]]
  matrix(rnorm(n=r*c), nrow=r, ncol=c)
  })
```

**Neuron**

A neuron is a basic unit in the DNN which is biologically inspired model of the human neuron. A single neuron performs weight and input multiplication and addition (FMA), which is as same as the linear regression in data science, and then FMA’s result is passed to the activation function. The commonly used activation functions include [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function), [ReLu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)), [Tanh](https://reference.wolfram.com/language/ref/Tanh.html) and Maxout. In this post, I will take the sigmoid  as activation function, $f(x) = \frac{1}{1+e^{-x}}$. For other types of activation function, you can refer [here](http://cs231n.github.io/neural-networks-1/#actfun).

```{r}
# Calculate activation function
sigmoid <- function(z){1.0/(1.0+exp(-z))}

# feedforward by matrix multiplication
feedforward <- function(a, biases, weights)
{
  for (f in 1:length(biases)){
    a <- matrix(a, nrow=length(a), ncol=1)
    b <- biases[[f]]
    w <- weights[[f]]
    w_a <- w%*%a
    b_broadcast <- matrix(b, nrow=dim(w_a)[1], ncol=dim(w_a)[-1])
    a <- sigmoid(w_a + b_broadcast)
  }
  a
}
```

**Layer**

– Input Layer

the input layer is relatively fixed with only 1 layer and the unit number is equivalent to the number of features in the input data.

-Hidden layers

Hidden layers are very various and it’s the core component in DNN. But in general,  more hidden layers are needed to capture desired patterns in case the problem is more complex (non-linear).

-Output Layer

The unit in output layer most commonly does not have an activation because it is usually taken to represent the class scores in classification and arbitrary real-valued numbers in regression. For classification, the number of output units matches the number of categories of prediction while there is only one output node for regression.

## Optimization

To reiterate, the loss function lets us quantify the quality of any particular set of weights W. The goal of optimization is to find W that minimizes the loss function. We will now motivate and slowly develop an approach to optimizing the loss function. For those of you coming to this class with previous experience, this section might seem odd since the working example we’ll use (the SVM loss) is a convex problem, but keep in mind that our goal is to eventually optimize Neural Networks where we can’t easily use any of the tools developed in the Convex Optimization literature.

**Stochastic Gradient Descent**

[Stochastic gradient descent (often shortened in SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent), also known as incremental gradient descent, is a stochastic approximation of the gradient descent optimization method for minimizing an objective function that is written as a sum of differentiable functions. In other words, SGD tries to find minima or maxima by iteration. We use stochastic gradient descent as our optimisation method:

```{r}
 SGD <- function(training_data, epochs, mini_batch_size, lr, C, sizes, num_layers, biases, weights,
                 verbose=FALSE, validation_data)
 {
   # Every epoch
   for (j in 1:epochs){
     # Stochastic mini-batch (shuffle data)
     training_data <- sample(training_data)
     # Partition set into mini-batches
     mini_batches <- split(training_data, 
                           ceiling(seq_along(training_data)/mini_batch_size))
     # Feed forward (and back) all mini-batches
     for (k in 1:length(mini_batches)) {
       # Update biases and weights
       res <- update_mini_batch(mini_batches[[k]], lr, C, sizes, num_layers, biases, weights)
       biases <- res[[1]]
       weights <- res[[-1]]
     }
   }
   # Return trained biases and weights
   list(biases, weights)
 }
```

In this example we use the cross-entropy loss function:
$$C = -\frac{1}{n}\sum_xy(x)\ln(a(x)) + (1 - y(x))\ln(1-a(x))$$
Where $a=\sigma(\sum_iw_ix_i + b) = \sigma(z)$.

```{r}
cost_delta <- function(method, z, a, y) {
  if (method=='ce'){return (a-y)}
}
```

then, update the weights after each mini-batch has been forward and backwards-propagated:

```{r}
# Go through mini_batch
for (i in 1:nmb){
  x <- mini_batch[[i]][[1]]
  y <- mini_batch[[i]][[-1]]
  # Back propogation will return delta
  # Backprop for each observation in mini-batch
  delta_nablas <- backprop(x, y, C, sizes, num_layers, biases, weights)
  delta_nabla_b <- delta_nablas[[1]]
  delta_nabla_w <- delta_nablas[[-1]]
  # Add on deltas to nabla
  nabla_b <- lapply(seq_along(biases),function(j)
    unlist(nabla_b[[j]])+unlist(delta_nabla_b[[j]]))
  nabla_w <- lapply(seq_along(weights),function(j)
    unlist(nabla_w[[j]])+unlist(delta_nabla_w[[j]]))
}
# After mini-batch has finished update biases and weights:
# i.e. weights = weights - (learning-rate/numbr in batch)*nabla_weights
# Opposite direction of gradient
weights <- lapply(seq_along(weights), function(j)
  unlist(weights[[j]])-(lr/nmb)*unlist(nabla_w[[j]]))
  biases <- lapply(seq_along(biases), function(j)
  unlist(biases[[j]])-(lr/nmb)*unlist(nabla_b[[j]]))
# Return
list(biases, weights)
```

**backpropagation**

Backpropagation is a beautifully local process. Every gate in a circuit diagram gets some inputs and can right away compute two things: 1. its output value and 2. the local gradient of its inputs with respect to its output value. Notice that the gates can do this completely independently without being aware of any of the details of the full circuit that they are embedded in. However, once the forward pass is over, during backpropagation the gate will eventually learn about the gradient of its output value on the final output of the entire circuit. Chain rule says that the gate should take that gradient and multiply it into every gradient it normally computes for all of its inputs.

```{r}
# Backwards (update gradient using errors)
# Last layer
delta <- cost_delta(method=C, z=zs[[length(zs)]], a=activations[[length(activations)]], y=y)
nabla_b_backprop[[length(nabla_b_backprop)]] <- delta
nabla_w_backprop[[length(nabla_w_backprop)]] <- delta %*% t(activations[[length(activations)-1]])
# Second to second-to-last-layer
# If no hidden-layer reduces to multinomial logit
if (num_layers > 2) {
  for (k in 2:(num_layers-1)) {
    sp <- sigmoid_prime(zs[[length(zs)-(k-1)]])
    delta <- (t(weights[[length(weights)-(k-2)]]) %*% delta) * sp
    nabla_b_backprop[[length(nabla_b_backprop)-(k-1)]] <- delta
    testyy <- t(activations[[length(activations)-k]])
    nabla_w_backprop[[length(nabla_w_backprop)-(k-1)]] <- delta %*% testyy
  }
}
return_nabla <- list(nabla_b_backprop, nabla_w_backprop)
return_nabla
```

## Training, Prediction and Evaluation

**Data preprocessing**

Before training the neural netword, we should first do some data preprocessing, such as standarise inputs to range(0,1) so that the data dimensions have approximately the same scale.

```{r}
# Function to standarise inputs to range(0, 1)
scalemax <- function(df)
{
  numeric_columns <- which(sapply(df, is.numeric))
  if (length(numeric_columns)){df[numeric_columns] <- lapply(df[numeric_columns], function(x){
    denom <- ifelse(max(x)==0, 1, max(x))
    x/denom
  })}
  df
}
```

Then, we should split the datase into training set and testing set.
```{r}
# Shuffle before splitting
if (shuffle_input) {all_data <- sample(all_data)}
# Split to training and test
tr_n <- round(length(all_data)*train_ratio)
# Return (training, testing)
list(all_data[c(1:tr_n)], all_data[-c(1:tr_n)])
```

There are some other data preprocessing measures on website [cs231n](http://cs231n.github.io/neural-networks-2/#datapre) for referrence.

**Training and Prediction**

We can train a neutral network through function neutralnetwork. The size of it can be set by using a num-array. For example, we can train a 2-layer network by setting "size = c(4,40,3)", which means the input layer has four nodes and the only one hidden layer has 40 nodes and the output layer has three nodes(class). Then, we can use feedforward function to predict on testing set.

```{r}
neuralnetwork <- function(sizes, training_data, epochs, mini_batch_size, lr, C,
                          verbose=FALSE, validation_data=training_data)

get_predictions <- function(test_X, biases, weights)
{
  lapply(c(1:length(test_X)), function(i) {
    which.max(feedforward(test_X[[i]], biases, weights))}
  )
}
```

**Evaluation**

We can caculate and print accuracy to evaluate the model. In addition, confusion matrix is also a good choice.

```{r}
# Accuracy
correct <- sum(mapply(function(x,y) x==y, pred, truths))
total <- length(testing_data)
print(correct/total)
# Confusion
res <- as.data.frame(cbind(t(as.data.frame(pred)), t(as.data.frame(truths))))
colnames(res) <- c("Prediction", "Truth")
table(as.vector(res$Prediction), as.vector(res$Truth))
```

## Example: Run on Iris Dataset

``` r
head(iris)
```

    ##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
    ## 1          5.1         3.5          1.4         0.2  setosa
    ## 2          4.9         3.0          1.4         0.2  setosa
    ## 3          4.7         3.2          1.3         0.2  setosa
    ## 4          4.6         3.1          1.5         0.2  setosa
    ## 5          5.0         3.6          1.4         0.2  setosa
    ## 6          5.4         3.9          1.7         0.4  setosa

``` r
train_test_split <- train_test_from_df(df = iris, predict_col_index = 5, train_ratio = 0.7)
training_data <- train_test_split[[1]]
testing_data <- train_test_split[[2]]

in_n <- length(training_data[[1]][[1]])
out_n <- length(training_data[[1]][[-1]])

# [4, 40, 3] 
trained_net <- neuralnetwork(
    c(in_n, 40, out_n),
    training_data=training_data,
    epochs=30, 
    mini_batch_size=10,
    lr=0.5,
    C='ce',
    verbose=TRUE,
    validation_data=testing_data
)
```

    ## Epoch:  1  complete[1] 0.6888889
    ## Epoch:  2  complete[1] 0.6222222
    ## Epoch:  3  complete[1] 0.6666667
    ## Epoch:  4  complete[1] 0.7333333
    ## Epoch:  5  complete[1] 0.7111111
    ## Epoch:  6  complete[1] 0.9555556
    ## Epoch:  7  complete[1] 0.7111111
    ## Epoch:  8  complete[1] 0.7111111
    ## Epoch:  9  complete[1] 0.6222222
    ## Epoch:  10  complete[1] 0.8444444
    ## Epoch:  11  complete[1] 0.9555556
    ## Epoch:  12  complete[1] 0.7777778
    ## Epoch:  13  complete[1] 0.7555556
    ## Epoch:  14  complete[1] 0.7333333
    ## Epoch:  15  complete[1] 0.6666667
    ## Epoch:  16  complete[1] 0.9333333
    ## Epoch:  17  complete[1] 0.8222222
    ## Epoch:  18  complete[1] 0.7333333
    ## Epoch:  19  complete[1] 0.9333333
    ## Epoch:  20  complete[1] 0.7111111
    ## Epoch:  21  complete[1] 0.8888889
    ## Epoch:  22  complete[1] 0.6666667
    ## Epoch:  23  complete[1] 0.8
    ## Epoch:  24  complete[1] 0.8
    ## Epoch:  25  complete[1] 0.9555556
    ## Epoch:  26  complete[1] 0.9555556
    ## Epoch:  27  complete[1] 0.9555556
    ## Epoch:  28  complete[1] 0.9111111
    ## Epoch:  29  complete[1] 0.6888889
    ## Epoch:  30  complete[1] 0.9555556
    ## Training complete in:  0.5934589Training complete

``` r
# Trained matricies:
biases <- trained_net[[1]]
weights <- trained_net[[-1]]

# Accuracy (train)
evaluate(training_data, biases, weights)  #0.971
```

    ## [1] 0.9714286

    ##    
    ##      1  2  3
    ##   1 35  0  0
    ##   2  0 32  2
    ##   3  0  1 35

``` r
# Accuracy (test)
evaluate(testing_data, biases, weights)  #0.956
```

    ## [1] 0.9555556

    ##    
    ##      1  2  3
    ##   1 15  0  0
    ##   2  0 17  2
    ##   3  0  0 11

## Summary

In this post, we have shown how to implement R multi neural network from scratch. But the code is only implemented the core concepts of DNN, and the reader can do further practices by:

- Solving regression problems by adjusting proper parameters
- Selecting various hidden layer size, activation function, loss function
- Do some research on weight initialization
- Choose more optimization functions
- From the output above, we can see the accuracy of testing_data is lower than that of training_data, why and how to solve it.
- Visualizing the network architecture, weights, and bias by R, an example in [here](https://beckmw.wordpress.com/tag/nnet/).

## Notes

1. The entire source code of this post in here

2. The PDF version of this post in here
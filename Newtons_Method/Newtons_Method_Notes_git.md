Newton’s Method for Optimization
================
Edgar Derricho

# Introduction:

Newton’s method is a powerful optimization technique that extends well
into logistic regression, particularly useful for maximum likelihood
estimation (MLE). Let us begin with the motivation:

# Motivation:

In logistic regression, our goal is to estimate the probability of a
binary outcome $y$(0 or 1) given a set of predictors $x$, using the
logistic function:
$$P(y=1|x)=\frac{1}{1+e^{-(\beta_{0}+\beta_{1}x_{1}+...+\beta_{n}x_{n})}}$$

To fit this model, we use MLE, seeking to find the parameters $\beta$
that maximize the likelihood of observing our data. Newton’s method with
its faster convergence rate, helps solve this by finding a local maximum
of the log-likelihood function.

# Newton’s Method for Optimization:

Newton’s Method is an iterative optimization technique that uses the
gradient and curvature (second derivative) of a function to quickly
converge to a solution. For logistic regression, this involves:</br>

- Gradient: The first derivative of the log-likelihood function
  concerning $\beta$</br>
- Hessian: The second derivative (or matrix of second partial
  derivatives) of the log-likelihood concerning $\beta$.</br>

In general, newton’s update step for a parameter $\beta$ is given
by:</br>
$$\beta^{new}=\beta^{old}-[H(\beta^{old})]^{-1} \nabla L(\beta^{old})$$

where:</br> \* $\nabla L(\beta)$ is the gradient of the
log-likelihood</br> \* $H(\beta)$ is the Hessian matrix of the
log-likelihood</br>

# Logistic Regression: Likelihood Function, Gradient and Hessian

For logistic regression, the log-likelihood function $L(\beta)$ for $n$
observation is:

$$L(\beta)=\sum^{n}_{i=1}(y_{i}log(p_{i})+(1-y_{i})log(1-p_{i}))$$

where $p_{i}=\frac{1}{1+e^{-\beta^{T}x_{i}}}$</br> </br>

The gradient of $L(\beta)$ with respect to $\beta$ is:

$$\nabla L(\beta)=X^{T}(y-p)$$

where:

- $X$ is the design matrix of predictors
- $y$ is the vector of observed outcomes
- $p$ is the vector of predicted probabilities, calculated as
  $p=\frac{1}{1+e^{-\beta^{T}x_{i}}}$</br>

The Hessian matrix $H(\beta)$ is:

$$H(\beta)=-X^{T}WX$$

where $W$ is a diagonal matrix with entries $W_{ii}=p_{i}(1-p_{i})$

# Implimentaiton in R:

``` r
# Sample Data:
set.seed(123)
n <- 100 # Number of Observations
x <- matrix(rnorm(n*2),n,2) # Two Predictors
beta_true <- c(0.5,-0.3)
y <- rbinom(n,1,plogis(x %*% beta_true))

# Adding Intercept Term:
X <- cbind(1,x) # Design Matrix with Intercept
beta <- rep(0, ncol(X)) # Initial Values for Beta Coefficient

# Newton's Method Parameters:
max.iter <- 100
tol <- 1e-6

# NM Loop:
for(i in 1:max.iter){
  p <- 1/(1 + exp(-X %*% beta)) # Predicted Probabilities
  gradient <- t(X) %*% (y-p) # Gradient of the Log-Likelihood
  W <- diag(as.vector(p * (1-p))) # Diagonal Matrix
  hessian <- -t(X) %*% W %*% X # Hessian Matrix
  
  # Newton's Update Step:
  beta_new <- beta - solve(hessian) %*% gradient
  
  # Check Convergence:
  if(sum(abs(beta_new - beta)) < tol){
    beta <- beta_new
    break
  }
  beta <- beta_new
  
}

cat("Estimated Coefficients:\n",beta)
```

    ## Estimated Coefficients:
    ##  0.02968842 0.2870145 -0.245101

# Explanation of code:

1.  **Data Generation**: Simulated binary outcomes and predictors

2.  **Initialization:** $X$ is the design matrix (including the
    intercept), nd $\beta$ is initialized as a zero vector.

3.  **Iterative Optimization**:

- Calculate predicted probabilities $p$
- Compute the gradient and Hessian of the log-likelihood
- Perform the Newton’s Update

4.  **Convergence Check**: The loop stops when the change in $\beta$
    falls below a tolerance level, indicating convergence

**This code gives the estimated coefficients, approximating those that
maximize the likelihood**

# Checking Model Performance:

Now, let us check for model accuracy. We will use a created dataset
where the x-values are from a sequence from 1 to 500 with added noise.
The response variable is whether or not the x-value is greater than the
rolling average with a window of $3$:

``` r
# Create Data:
window <- 3 # Moving Average Window
n <- 500 # Data Points
x_val <- seq(1,n,by=1) + rnorm(n,0,n) # Value + Noise
moving_average <- roll_mean(x_val,window) # Moving Average   
y <- ifelse(x_val > moving_average,1,0) # Binary Response

# Data Frame of X and Y Values:
xValues <- x_val[window:n]
moving_average <- moving_average[window:n]
yValues <- y[window:n]
data <- data.frame(xValues,moving_average,yValues)
head(data)
```

    ##      xValues moving_average yValues
    ## 1 -168.95862      -211.5661       1
    ## 2   49.24832      -132.8828       1
    ## 3  804.25439       228.1814       1
    ## 4  -38.28256       271.7401       0
    ## 5  547.39975       437.7905       1
    ## 6  323.37706       277.4981       1

Graph the scatterplot:

``` r
pch <- 9 # pch Value

# Plot
plot(data$xValues,
     col = ifelse(y == 1, "blue","red"),
     xlab = "Index",
     ylab = "Values",
     main = "Scatterplot of Values",
     pch = pch
     )

# Legend for Plot
legend("topleft",
       legend = c("Y > Moving Avg.","Y < Moving Avg."),
       col = c("blue","red"),
       pch = pch,
       )
```

![](Newtons_Method_Notes_git_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

![](Newtons_Method_Notes_git_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Define the initial parameters and compute initial log-likelihood

``` r
# Create Matrices for X and Y:
X <- as.matrix(cbind(1,data$xValues))
Y <- as.matrix(data$yValues)

# Initialize beta at zeros:
beta_init <- rep(0,ncol(X))

# Logistic Function:
logistic <- function(z){
  1 / ( 1 +exp(-z))
}

# Coefficient Vector
beta_vec <- c()

# Log-Likelihood Function:
log_likelihood <- function(Y, X, beta){
  p <- logistic(X %*% beta)
  sum(Y * log(p) + (1 - Y) * log(1 - p))
}

# Compute initial Log-Likelihood:
initial_ll <- log_likelihood(Y, X, beta_init)
cat("Initial Log Likelihood: ", initial_ll)
```

    ## Initial Log Likelihood:  -345.1873

Apply Newton’s Method to Optimize Parameters

``` r
# Newton's Method parameters
max_iter <- 100
tol <- 1e-6
beta <- beta_init

# Beta Vector
beta_vec <- c()

# Newton's Method implementation
for (i in 1:max_iter) {
  p <- logistic(X %*% beta)
  gradient <- t(X) %*% (Y - p)
  W <- diag(as.vector(p * (1 - p)))
  hessian <- -t(X) %*% W %*% X
  
  # Regularize Hessian to ensure it's invertible
  lambda <- 1e-5
  hessian_reg <- hessian - lambda * diag(ncol(hessian))
  
  # Update beta
  beta_new <- beta - solve(hessian_reg) %*% gradient
  
  # Update Beta Vector
  beta_vec <- append(beta_vec,beta_new)
  
  # Check convergence
  if (sum(abs(beta_new - beta)) < tol) {
    beta <- beta_new
    break
  }
  
  beta <- beta_new
}

# Beta Matrix:
b.mat <- matrix(beta_vec,ncol = 2,byrow = T)

# Compute final log-likelihood:
optimized_log_likelihood <- log_likelihood(Y, X, beta)

# Initial and Optimized Log Likelihood:
cat(
    "Initial Log Likelihood: ", initial_ll, "\n",
    "Optimized Log-Likelihood (after Newton's Method):", optimized_log_likelihood
    )
```

    ## Initial Log Likelihood:  -345.1873 
    ##  Optimized Log-Likelihood (after Newton's Method): -215.751

Accuracy Improvement:

``` r
# Predictions and accuracy at initial parameters
initial_pred <- ifelse(logistic(X %*% beta_init) > 0.5, 1, 0)
initial_accuracy <- mean(initial_pred == Y)

# Predictions and accuracy after Newton's Method
optimized_pred <- ifelse(logistic(X %*% beta) > 0.5, 1, 0)
optimized_accuracy <- mean(optimized_pred == Y)

# Output Both Accuracy Percentages:
cat(
    "Initial Accuracy:", initial_accuracy, "\n",
    "Optimized Accuracy:", optimized_accuracy
    )
```

    ## Initial Accuracy: 0.4759036 
    ##  Optimized Accuracy: 0.7771084

With an increase in accuracy and a decrease in likelihood means Newton’s
method is beneficial for improving model accuracy.

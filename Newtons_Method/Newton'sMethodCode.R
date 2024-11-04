# ---------------------------------------------------------------------
# This is an implementation of Newton's method for optimization of
# logistic regression coefficients
# ---------------------------------------------------------------------

# Libraries

#install.packages("roll")
#install.packages("tidyverse")

library(roll)
library(tidyverse)

#------------------------------------------------------------------------
#------------------------ Implementation --------------------------------
#-------------------------------------------------------------------------

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

#------------------------------------------------------------------------
#---------------------- Checking Model Performance -----------------------
#-------------------------------------------------------------------------

#We will use a created data set where the x-values are from a sequence 
# from 1 to 500 with added noise. The response variable is whether or 
# not the x-value is greater than the rolling average with a window of 3:

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

#------------------------------------------------------------
#Graph the Scatterplot

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

#----------------------------------------------------------
# Define the initial parameters and compute initial log-likelihood:

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

#--------------------------------------------
#Apply Newton's Method to Optimize Parameters:

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
## Initial Log Likelihood:  -345.1873 
##  Optimized Log-Likelihood (after Newton's Method): -215.751

# -----------------------------------------------------------------
# Accuracy Improvement:
  
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
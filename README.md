Coursera Machine Learning - Andrew Ng
=====================================


# Linear Regression with One Variable
* Univariate (Single Variable) Regression Model
* Model: h(x) = theta0 + theta1(x) 


### Feature Scaling (Gradient Decent)
* Get value of all the features in the range of -1 >= 0 <= +1
	- X0 =  (Value - AvgVal) / (max - min) 
	- X0 = (#Bedrooms - Avg # Of Bedrooms) / (Max Bedrooms - Min Bedrooms)
	
	```
		mu = mean(X);
		X_norm = bsxfun(@minus, X, mu);

		sigma = std(X_norm);
		X_norm = bsxfun(@rdivide, X_norm, sigma);
	```


### Learning Rate (Gradient Decent)
* Choose learning rate small but not too small
	- If learning rate too small the convergence will take a lot of iterations and it will take a lot of time to find the minimum
	- If learning rate too big the convergence may never happen as the iteration may overshoot.
	- chose values like 0.001, 0.01, 0.1, 1 and increase.
	- Plot of J(theta) should go down not up.


### Feature Selection
* Create your own features if necessary
	- for predicting house price if you have features length and breadth you can create you own feature area as length * breadth

* Try different polynomial functions of existing features 
	- If you size as feature add new feature size^2
	- If using polynomial functions feature scaling becomes very important


### Gradient Decent vs Normal Equation
* Use Gradient Decent if you have too many features (n >= 1000)
* Normal Equation not good for large (n > 1000) as it may slow down.



### Linear Regression (Normal Equation)
```
pinv(X'*X)*X'*y
```


# Classification

### Logistic Regression

### Sigmoid (Logistic) function

```
g (θ' * x)
z = θ' * x
g (z) = 1 / 1 + exp^-z
```

* h(x) = P(y = 1|X; θ)
* P(y = 1|X; θ) + P(y = 0|X; θ) = 1


### Cost function

### Gradient

### Regularization (To avoid overfitting)

#### Keep all the features, but reduce magnitude/values of parameters .
- Works well when we have a lot of features, each of which contributes a bit to predicting .

#### What if λ is set to an extremely large value (perhaps for too large for our problem, say  λ - 10 <sup>10</sup> )?
- Algorithm works fine; setting to be very large can’t hurt it
- Algortihm fails to eliminate overfitting.
- Algorithm results in underfitting. (Fails to fit even training data well).
- Gradient descent will fail to converge.

### Regularized Cost function

### Regularized Gradient






# Octave Functions
```
* Matrix Inverse: pinv(X) - psuedo inverse function
* Matrix Transpose: X'
* Normal Equation: pinv(X'*X)*X'*y
* ones(3,3) - create 3x3 mat with all 1
* 2*ones(3,3) - create 3x3 mat with all 2
* a = rand(3,3) - create mat with rand numbers
* hist(a) - create histogram
* eye(4,4) - create identity matx
* size(A)
* [m, n] = size(A) - get size of matrix in m,n
* load(filename)
* clear varname - deletes the var
* who - list all var
* whos - list all var with details
* theta(2:end,1) - all the rows in theta starting from 2-to-end at col-1
```














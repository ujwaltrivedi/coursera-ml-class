function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
gradStore = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



[m, n] = size(X);
z = X * theta;
hx = sigmoid(z);

% calculate regularization on all theta > 1
REG = (lambda / (2 * m)) * sum(theta(2:end) .^ 2);

J = ( (1/m) * sum(-y .* log(hx) - (1 - y) .* log(1 .- hx)) ) + REG;



for ts = 1:length(gradStore)

  if (ts <= 1)
    REGD = 0;
  else
    REGD = (lambda / m) * theta(ts);
  endif
  
  gradStore(ts) = (1/m) * sum((hx .- y) .* X(:,ts)) + REGD;

end

grad = gradStore;


% =============================================================

end

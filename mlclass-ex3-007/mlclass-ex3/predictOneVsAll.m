function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       


% one-vs-all
% 10 classes
% x = 5000,401
% z = 5000,10
% all_theta = 10,401

z = zeros(size(X, 1), num_labels);
hx = z;


% test all X for each class from 1 to 10
% put the results in z(:,1 to 10)
for ts = 1:num_labels
  z(:,ts) = X * all_theta(ts,:)';
  hx(:,ts) = sigmoid(z(:,ts));
end

  
% get the index of class with max probability and put it in the "p"
for ts = 1:m(1)
   [x,ix] = max(hx(ts,:));    
   p(ts) = ix;    
end



% =========================================================================

end

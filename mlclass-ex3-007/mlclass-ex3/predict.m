function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% add column with all 1's
X = [ones(m, 1) X];

% calculte layer-1, input layer
layer1_z      = X * Theta1';
layer1_output  = sigmoid(layer1_z);

% add column with all 1's
layer1_output = [ones(size(layer1_output,1),1) layer1_output];


% calculte layer-2, hidden layer
layer2_z = layer1_output * Theta2'
layer2_output  = sigmoid(layer2_z);


% predict 
for ts = 1:m
   [x,ix] = max(layer2_output(ts,:));    
   p(ts) = ix;    
end




% =========================================================================


end

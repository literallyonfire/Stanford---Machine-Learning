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


%Add a column of 5000x1 bias units into X
X = [ones(m,1) X];

%Feed Forward with theta1 and theta2 Matrices
h1 = sigmoid(X * Theta1');
h1 = [ones(m,1) h1];
h2 = sigmoid(h1 * Theta2');

%get the index of the highest probable class of examples
[val,idx] = max(h2, [], 2);

%return the answer
p = idx;


% =========================================================================


end

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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%Theta with theta(1) = 0 to help with regularization and as theta(1) is irrevelant with regularization
theta_dup = theta;
theta_dup(1) = 0;

%Calculates hypothesis sigmoid function 
z = X * theta;
h = sigmoid(z);

%Logistic regression cost function with regularization added
J = -1/m .* sum(y.*log(h) + (1-y).*log(1-h)) + lambda/(2*m) .* sum(theta_dup .^ 2);

%Calculates gradient with regularization
grad = 1/m .* X' * (h - y) + lambda/m * theta_dup;

% =============================================================

end

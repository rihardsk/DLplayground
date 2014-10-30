function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = logical(full(sparse(labels, 1:numCases, 1)));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

m = size(data, 2);

raw = exp(data' * theta');
h = raw ./ repmat(sum(raw, 2), 1, numClasses);

cost = - sum(log(h(groundTruth')(:))) / m + lambda * theta(:)' * theta(:) / 2;

thetagrad = - (data * (groundTruth' - h))' / m + lambda * theta;




% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
fflush(stdout);
end

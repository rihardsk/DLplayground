function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = logical(full(sparse(labels, 1:M, 1)));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%
m = M;

net = cell(size(stack));
net{1}.a = data';
l = numel(stack) + 1; % autoencoder layer count (without softmax layer)
% forward-propagation
for d = 2:l
    net{d}.a = sigmoid(net{d - 1}.a * stack{d - 1}.w' + stack{d - 1}.b');
end

numclases = size(softmaxTheta, 1);

%softmax predict
raw = exp(net{end}.a * softmaxTheta');
h = raw ./ repmat(sum(raw, 2), 1, numclases);

% šī ir softmax cost funkcija
cost = - sum(log(h(groundTruth')(:))) / m + lambda * softmaxTheta(:)' * softmaxTheta(:) / 2;

% cost funkcijas atvasinājums pēc softmax Theta parametriem
softmaxThetaGrad = softmaxThetaGrad - (net{end}.a' * (groundTruth' - h))' / m + lambda * softmaxTheta;

% pēdējais SAA līmenis tiek sasaistīts ar softmax līmeni (atvasinājuma ziņā)
%  = - (softmax parciālie atvasinājumi) .* (sigm. f-jas atvasinājuma vērtība uz pēdējai līmeņa izvada)
net{l}.d = - (softmaxTheta' * (groundTruth - h'))' .* (net{l}.a .* (1 - net{l}.a));
stackgrad{l - 1}.w = stackgrad{l - 1}.w + net{l}.d' * net{l - 1}.a / m;
stackgrad{l - 1}.b = stackgrad{l - 1}.b + sum(net{l}.d)' / m;
% tālāk parastais backpropagation
for i = l-1:-1:2
    net{i}.d = net{i + 1}.d * stack{i}.w .* (net{i}.a .* (1 - net{i}.a));
    stackgrad{i - 1}.w = stackgrad{i - 1}.w + net{i}.d' * net{i - 1}.a / m;
    stackgrad{i - 1}.b = stackgrad{i - 1}.b + sum(net{i}.d)' / m;
end



% sparsity param netiek ņemts vērā šoreiz
% d3 = dif .* (a3 .* (1 - a3)); %  * -1
% d2 = (d3 * W2 + beta * (-sparsityParam./aavg + (1 - sparsityParam)./(1 - aavg))) .* (a2 .* (1 - a2));
%      ---^---   ------------------^----------------------------------------         ------^---------
%  no lambda param.      no sparsity ierobežojuma                                    funkcijas atvas.



% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

fflush(stdout);
end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

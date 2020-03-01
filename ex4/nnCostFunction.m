function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% agrego el bias
a1 = [ones(m, 1), X];
z2 = a1*Theta1.';
a2 = sigmoid(z2);
% agrego el bias
a2 = [ones(m, 1), a2];
z3 = a2*Theta2.';
h = sigmoid(z3);

% en cada fila tengo un label correspondiente a un ejemplo
% h también tiene los ejemplos en filas
Y = zeros(m, num_labels);
for i = 1:m
    l = y(i);
    Y(i, l) = 1;
end

J = 0;
for k = 1:num_labels
    for i = 1:m
        J = J + -Y(i, k)*log(h(i, k)) - (1 - Y(i, k))*log(1 - h(i, k));
        % término de regularización
        %r = 
    end
end

J = (1/m)*J;

% regularización
% omito la 1er columna
t1 = Theta1(:, 2:end);
t2 = Theta2(:, 2:end);
r = sum(sum(t1.^2)) + sum(sum(t2.^2));

J = J + (lambda/(2*m))*r;

%%%%%%%%%%% parte 2 %%%%%%%%%%%

% para qué tengo que hacer el ff otra vez??

% cada ejemplo es una fila
% inicializo las variables, porque ya las usé
a1 = 0;
z2 = 0;
a2 = 0;
z3 = 0;
h = 0;

D1 = 0;
D2 = 0;
for i = 1:m
    
    % agrego el bias
    a1 = [1, X(i, :)];
    a1 = a1.';
    
    % calculo a2, a3
    z2 = Theta1*a1;
    a2 = [1; sigmoid(z2)];
    z3 = Theta2*a2;
    h = sigmoid(z3);
    
    d3 = h - Y(i, :).';
    % hay que agregarle un término a z2 para que se pueda hacer la cuenta,
    % y después descartar ese primer término
    d2 = (Theta2.'*d3).*sigmoidGradient([1; z2]);
    d2 = d2(2:end);
    
    % acumulo el gradiente
    D1 = D1 + d2*a1.';
    D2 = D2 + d3*a2.';
end



Theta1_grad = D1/m;
Theta2_grad = D2/m;

% primero calculo el t de reg, y después le mato la 1er columna
reg = Theta1*lambda/m;
reg(:, 1) = zeros(size(Theta1, 1), 1);
Theta1_grad = Theta1_grad + reg;

reg = Theta2*lambda/m;
reg(:, 1) = zeros(size(Theta2, 1), 1);
Theta2_grad = Theta2_grad + reg;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

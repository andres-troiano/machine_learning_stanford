data = load('ex1data1.txt'); % read comma separated data
X = data(:, 1);
y = data(:, 2);

m = length(X); % number of training examples
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
iterations = 1500;
alpha = 0.01;

m = length(y); % number of training examples

J = 0;

h = X*theta;

% j = 1
aux = sum((h - y).*X(:, 1));
theta_1 = theta(1) - (alpha/m)*aux;

% j = 2
aux = sum((h - y).*X(:, 2));
theta_2 = theta(2) - (alpha/m)*aux;
    
theta = [theta_1; theta_2];
J = (1/(2*m))*sum((h - y).^2);

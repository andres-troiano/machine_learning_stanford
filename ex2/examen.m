x = [9; 7; 34];

theta_1 = [[1, -1.5, 3.7]; [1, 5.1, 2.3]];
theta_1_var = [[1, 5.1, 2.3]; [1, -1.5, 3.7]];
theta_2 = [1, 0.6, -0.8];
theta_2_var = [1, -0.8, 0.6];

z = theta_1*x;
a2 = sigmoid(z);
a2 = [1; a2];
z = theta_2*a2;
a3 = sigmoid(z)

z = theta_1_var*x;
a2 = sigmoid(z);
a2 = [1; a2];
z = theta_2_var*a2;
a3 = sigmoid(z)
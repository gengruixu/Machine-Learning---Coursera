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
% y_new = OutputLayer(y,num_labels); %5000*10
% X = [ones(m,1) X];  %5000*401
% %-------------------FP-----------------------------------
% z_2 = X*Theta1';    %5000*25
% a_2 = sigmoid(z_2); %5000*25
% a_2 = [ones(m,1) a_2];  %5000*26
% z_3 = a_2*Theta2';  %5000*10
% a_3 = sigmoid(z_3); %5000*10
% for t=1:m     %For each sample
%     for k=1:num_labels    %For each element in the last column
%         J = J - (1/m)*(y_new(t,k)*log(a_3(t,k)) + ((1-y_new(t,k))*log(1-a_3(t,k))));
%     end
% end
% J = J + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
% %         J = J -(1/m)*sum(sum(y'*log(sigmoid(z_3))+(1-y)'*log(1-sigmoid(z_3))))+(lambda/(2*m))*(sum(sum(Theta1.^2))+sum(sum(Theta2.^2)));
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
a1 = [ones(m,1) X];  
  
z2 = a1*Theta1';  
a2 = sigmoid(z2);  
m1 = size(a2,1);  
a2 = [ones(m1,1) a2];  
  
z3 = a2*Theta2';  
a3 = sigmoid(z3);  
  
%temp = log(a3);  
  
for i=1:num_labels;  
    ynew(:,i) = y==i; %5000x10  
end  
  
J = 1/m*sum(sum(-ynew.*log(a3)-(1-ynew).*log(1-a3)))...  
    +lambda/(2*m)*(sum(sum(Theta1(1:hidden_layer_size,2:1+input_layer_size).^2)) ...  
    + sum(sum(Theta2(1:num_labels,2:1+hidden_layer_size).^2)));  
  
Delta1 = zeros(size(Theta1)); %25x401  
Delta2 = zeros(size(Theta2)); %10x26  
for i=1:m;  
    a1 = X(i,:); %a1 is 1x400  
    a1 =[1 a1]; %1x401  
    z2 = a1*Theta1';%1x25  
    a2 = sigmoid(z2);  
    m1 = size(a2,1);  
    a2 = [ones(m1,1) a2]; %1x26  
  
    z3 = a2*Theta2'; %1x10  
    a3 = sigmoid(z3); %1x10  
      
    delta_3 = a3 - ynew(i,:); %1x10  
    temp = delta_3*Theta2; %1x26  
    delta_2 = temp(2:end).*sigmoidGradient(z2); %1x25  
      
    Delta1 = Delta1 + delta_2'*a1;  
    Delta2 = Delta2 + delta_3'*a2;  
end  
  
m1 = size(Theta1,1);  
m2 = size(Theta2,1);  
Thet1 = [zeros(m1,1) Theta1(:,2:end)];  
Thet2 = [zeros(m2,1) Theta2(:,2:end)];  
  
Theta1_grad = Delta1/m +lambda/m*Thet1; %25x401  
Theta2_grad = Delta2/m +lambda/m*Thet2; %10x26     


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

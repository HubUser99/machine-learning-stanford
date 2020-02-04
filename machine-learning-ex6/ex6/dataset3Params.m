function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

range1 = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
range2 = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
errors = zeros(8, 8);

for i = 1:length(range1)
  for j = 1:length(range2)
    disp(i);
    disp(j);
    model= svmTrain(X, y, range1(i), @(x1, x2) gaussianKernel(x1, x2, range2(j)));
    predictions = svmPredict(model, Xval);
    errors(i,j) = mean(double(predictions ~= yval));
  endfor
endfor

[_, CIndex] =  min(min(errors, [], 2));
[_, sigmaIndex] =  min(min(errors, [], 1));

disp(CIndex);
disp(sigmaIndex);

disp(range1(CIndex));
disp(range2(sigmaIndex));

C = range1(CIndex);
sigma = range2(sigmaIndex);

% =========================================================================

end

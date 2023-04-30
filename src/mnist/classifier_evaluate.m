function classifier_evaluate(class_estimated,class_true)
%CLASSIFIER_EVALUATE Calculates and dislays the error rate and confusion
%matrix given a class estimate and class ground truth

num_samples = length(class_estimated);

% Find confusion matrix and error rate for the test set.
mtrx_confusion = calc_confusion_matrix(class_true(1:num_samples), class_estimated);
is_equal = class_estimated == class_true(1:num_samples);
num_correct = sum(is_equal);
error_rate = (num_samples-num_correct)/num_samples * 100;

% Display
disp(strcat("The error rate for the classifier is ", num2str(error_rate), "%."));

figure;
hmo = heatmap({'0', '1', '2,' '3', '4', '5', '6', '7', '8', '9'}, {'0', '1', '2,' '3', '4', '5', '6', '7', '8', '9'}, mtrx_confusion);
hmo.Title = "Classifier confusion matrix";
hmo.XLabel = "True value";
hmo.YLabel = "Classification";
hmo.ColorScaling = 'log';
end


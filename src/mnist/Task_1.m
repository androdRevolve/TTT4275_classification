%% Design NN-based classifier using Euclidian distance

% Use a subset of num_samples samples as reference. For each feature vector x,
% find the training reference with the smallest distance and classify as
% that target
%% Init
num_references = 1000;
num_samples = 10000;
%% Run classification using the first num_references references and the first num_samples test features
tic;

mtrx_references = trainv(1:num_references,:)';
vec_targets = trainlab(1:num_references);

classes = zeros(num_samples,1);
for test_samp = 1:num_samples
    x_test = testv(test_samp,:)';
    mtrx_dist = calc_distance_euclidian(x_test,mtrx_references);
    distances = diag(mtrx_dist);
    [dist_min,ind_min] = min(distances);
    classes(test_samp) = vec_targets(ind_min);
end
toc;
%% Evaluate performance

% Find confusion matrix and error rate for the test set.
% The data sets should preferably be split up into chunks of images (for example 1000) in order to 
% a) avoid too big distance matrixes 
% b) avoid using excessive time (as when classifying a single image at a time)
mtrx_confusion = calc_confusion_matrix(testlab(1:num_samples), classes);
is_equal = classes == testlab(1:num_samples);
num_correct = sum(is_equal);
error_rate = (num_samples-num_correct)/num_samples * 100;

% Display
disp(strcat("The error rate for the unclustered NN is ", num2str(error_rate), "%."));

figure;
hmo = heatmap({'0', '1', '2,' '3', '4', '5', '6', '7', '8', '9'}, {'0', '1', '2,' '3', '4', '5', '6', '7', '8', '9'}, mtrx_confusion);
hmo.Title = "Unclustered NN confusion matrix";
hmo.XLabel = "True value";
hmo.YLabel = "Classification";
hmo.ColorScaling = 'log';

%% Plot some misclassified pictures
% Plot the first 5 misclassified pictures
num_plotted = 0;
i = 5000;
while num_plotted < 5
    if(~is_equal(i))
        % Convert picture from vector to matrix
        mtrx_pic = zeros(col_size,row_size); mtrx_pic(:) = testv(i,:);
        image(mtrx_pic');
         num_plotted = num_plotted+1;
    end
    i = i+1;
end

%% Plot some correctly classified pixtures
% Plot the first 5 correctly classified pixture
num_plotted = 0;
i = 5000;
while num_plotted < 5
    if(is_equal(i))
        % Convert picture from vector to matrix
        mtrx_pic = zeros(col_size,row_size); mtrx_pic(:) = testv(i,:);
        image(mtrx_pic');
        num_plotted = num_plotted+1;
    end
    i = i+1;
end

%% Conclusion
% The unclustered NN classifier has an error rate of 13.1% on the whole
% dataset (10k samples) using 1k references. The error rate is quite
% sensitive to the number of references, exemplified by an error rate of
% 9.72% When using 2k references.
% The runtime of the classifier is very long (135.45 s) when
% using 1000 references. When increasing the number of references to 2k,
% the runtime is 474.96 s.

% In particular, the classifier struggles with the
% distinction between the class pairs (4,9), (3,5), (8,5). Which makes sense as there is
% significant overlap in the shapes of these digits, and variation in
% handwriting can decide the margins.

disp('..done');



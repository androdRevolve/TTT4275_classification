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
classifier_evaluate(classes, testlab);

disp('..done');



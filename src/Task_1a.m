%% Design NN-based classifier using Euclidian distance

% Use a subset of num_samples samples as reference. For each feature vector x,
% find the training reference with the smallest distance and classify as
% that target
%% Init
num_references = 2000;
num_samples = 1000;
%% Run classification using the first num_references references and the first num_samples test features
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

%% Evaluate performance
is_equal = classes == testlab(1:num_samples);
num_correct = sum(is_equal);
error_rate = (num_samples-num_correct)/num_samples * 100;

%% Find confusion matrix and error rate for the test set.
% The data sets should preferably be split up into chunks of images (for example 1000) in order to 
% a) avoid too big distance matrixes 
% b) avoid using excessive time (as when classifying a single image at a time)


%% Plot some misclassified pictures
% Plot the first 5 misclassified pictures
num_plotted = 0;
i = 100;
while num_plotted < 5
    if(~is_equal(i))
        % Convert picture from vector to matrix
        mtrx_pic = zeros(pic_height,pic_width); mtrx_pic(:) = testv(i,:);
        image(mtrx_pic');
        num_plotted = num_plotted+1;
    end
    i = i+1;
end


disp('..done');



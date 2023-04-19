%% Task 2
% Use clustering to produce a small(er) set of templates for each class.
N_clusters = 64;
N_vectors_per_class = 6000;
N_classes = 10;

%% a)  Perform clustering of the 6000 training vectors for each class into M = 64 clusters.
% Separate training data into matrix based on class

load('class_clusters.mat');
% arr_class_clusters = cell(1,10);
% for i = 1:length(trainlab)
%     arr_class_clusters{trainlab(i)+1} = [arr_class_clusters{trainlab(i)+1}; trainv(i,:)];
% end

load('arr_clusters.mat');
% arr_clusters = cell(1,10);
% for i =1:length(arr_clusters)
%     [idx,C] = kmeans(arr_class_clusters{i}, N_clusters);
%     arr_clusters{i} = C;
% end

%% Perform classification
% Run classification on one class, find the
% minimum distances for all feature vectors

for i = 1:N_classes
    distances = dist(arr_clusters{i}, testv');
    min_distances(i,:) = min(distances,[],1);
end

% Classify by choosing the class with the minimum distance
[~,class] = min(min_distances,[],1);
class = class'-1;

%% b) Find the confusion matrix and the error rate for the NN classifier using these M = 64
% templates pr class. Comment on the processing time and the performance relatively to
% using all training vectors as templates.

% Error rate
is_equal = class == testlab;
num_correct = sum(is_equal);
num_errors = length(testlab)-num_correct;
error_rate = num_errors / length(testlab) * 100;

% Confusion matrix
mtrx_confusion = calc_confusion_matrix(testlab, class);

% Display
disp(strcat("The error rate is ", num2str(error_rate), "%."));
hmo = heatmap({'0', '1', '2,' '3', '4', '5', '6', '7', '8', '9'}, {'0', '1', '2,' '3', '4', '5', '6', '7', '8', '9'}, mtrx_confusion);
hmo.Title = "NN confusion matrix";
hmo.XLabel = "True value";
hmo.YLabel = "Classification";

%% c) Now design a KNN classifier with K=7. Find the confusion matrix and the error rate and
% compare to the two other systems.


disp('..done');

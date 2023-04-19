%% Task 2
% Use clustering to produce a small(er) set of templates for each class.
N_clusters = 64;
N_classes = 10;

%% a)  Perform clustering of the 6000 training vectors for each class into M = 64 clusters.
% Separate training data into matrix based on class
tic;

load('class_clusters.mat');
% arr_class_clusters = cell(1,10);
% for i = 1:length(trainlab)
%     arr_class_clusters{trainlab(i)+1}(end+1,:) = trainv(i,:);
% end

load('arr_clusters.mat');
% arr_clusters = cell(1,10);
% for i =1:length(arr_clusters)
%     [idx,C] = kmeans(arr_class_clusters{i}, N_clusters);
%     arr_clusters{i} = C;
% end
toc;


%% Perform classification
% Run classification on one class, find the
% minimum distances for all feature vectors
tic;
for i = 1:N_classes
    distances = dist(arr_clusters{i}, testv');
    min_distances(i,:) = min(distances,[],1);
end

% Classify by choosing the class with the minimum distance
[~,class] = min(min_distances,[],1);
class = class'-1;
toc;

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
disp(strcat("The error rate for the clustered NN is ", num2str(error_rate), "%."));

figure;
heatmap_NN = heatmap({'0', '1', '2,' '3', '4', '5', '6', '7', '8', '9'}, {'0', '1', '2,' '3', '4', '5', '6', '7', '8', '9'}, mtrx_confusion);
heatmap_NN.Title = "NN confusion matrix";
heatmap_NN.XLabel = "True value";
heatmap_NN.YLabel = "Classification";
heatmap_NN.ColorScaling = 'log';

%% c) Now design a KNN classifier with K=7. Find the confusion matrix and the error rate and
% compare to the two other systems.
tic;
K_knn = 7;

% Collect all distances in a single matrix
distances = zeros(N_classes*N_clusters, length(testlab));
for i = 1:N_classes
    distances(N_clusters*(i-1)+1:N_clusters*i,:) = dist(arr_clusters{i}, testv');
end

%% Classify these neighbors from their indices
% Identify the K nearest neighbours of each feature vector
[~,I] = mink(distances, K_knn, 1);
classes_neighbors = floorDiv(I,N_clusters+1);

classes = zeros(length(testlab), 1);
% Majority vote; identify the class with the highest number of neighbors,
% if two classes get equal amount of votes, choose the closest neighbor
for i=1:length(testlab)
    % Determine the count of each class
    [votes, class_val] = groupcounts(classes_neighbors(:,i));
    % Find the maximum votes
    [max_votes, ind] = max(votes);
    % Find number of contestants
    is_contestant = votes == max_votes;
    N_contestants = sum(is_contestant);
    if N_contestants > 1
        % Determine what classes are contestants
        contestants = class_val(is_contestant);
        % Find the shortest distance to each of the classes
        contestant_distance = zeros(N_contestants, 1);
        for j=1:N_contestants
            contestant_distance(j) = min(dist(arr_clusters{contestants(j)+1}, testv(i,:)'));
        end
        % Classify as the class with the lowest distance
        [~, ind] = min(contestant_distance);
        classes(i) = contestants(ind);
    else
        classes(i) = class_val(ind);
    end
end
toc;
%% Evaluate performance

% Error rate
is_equal = classes == testlab;
num_correct = sum(is_equal);
num_errors = length(testlab)-num_correct;
error_rate = num_errors / length(testlab) * 100;

% Confusion matrix
mtrx_confusion = calc_confusion_matrix(testlab, classes);

% Display
disp(strcat("The error rate for the clustered KNN classifier is ", num2str(error_rate), "%."));
figure;
heatmap_KNN = heatmap({'0', '1', '2,' '3', '4', '5', '6', '7', '8', '9'}, {'0', '1', '2,' '3', '4', '5', '6', '7', '8', '9'}, mtrx_confusion);
heatmap_KNN.Title = "KNN confusion matrix";
heatmap_KNN.XLabel = "True value";
heatmap_KNN.YLabel = "Classification";
heatmap_KNN.ColorScaling = 'log';

%% Conclusion
% Interestingly, the NN classifier outperforms the KNN classifier. Looking
% at the results (define test = (class == testlab & classes ~= testlab)) it
% appears that the reason is that the KNN classifier sometimes gets
% "infiltrated" by wrong neighbors, e.g. "9" is correct and closest, but
% "9" gets outvoted by "7". The conclusion: for this dataset, the
% "MAP"-classifier outperforms the KNN; we should rather trust the single
% closest neighbor than the K closest.

% The KNN classifier outperforms the NN classifier without clustering
% implemented in task 1.

% Clustering has a big impact on the runtime of the classification. With
% clustering, some time is spent on the clustering algorithm that
% initializes the references. When the references are done, the computation
% is much faster for each new feature x, as the total number of references
% is much lower (640 vs 1k in the default case).

% Timing:
% 64 clusters
% Reference sorting and clustering: 98.30 seconds 
% Classification: 11.42 seconds (NN), 12.1 seconds (KNN)

% 32 clusters
% Reference sorting and clustering: 99.36 seconds
% Classification: 5.85 seconds (NN), 6.77 seconds (KNN)

% Error rates:
% 64 clusters: 4.6% (NN), 6.02% (KNN)
% 32 clusters: 5.26% (NN), 8.95% (KNN)


disp('..done');

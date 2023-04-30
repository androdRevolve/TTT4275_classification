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

%% Evaluate
classifier_evaluate(class,testlab);

%% c) Now design a KNN classifier with K=7
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
classes_neighbors = floorDiv(I-1,N_clusters);

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
classifier_evaluate(classes, testlab);

disp('..done');

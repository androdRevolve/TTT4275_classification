clear all
close all

plot_histograms = true;
remove_sepal_width = true;
remove_sepal_length = true;
remove_petal_length = true;
%trainSamples: indeces of samples used for training
trainSamples = 1:30;
%testSamples: indeces of samples used for evaluation
testSamples = 31:50;
%training_mode: "softmax" for softmax activation function and cross-entropy
%loss function, "sigmoid" for sigmoid activation function and L2 error loss
%function
training_mode = "oftmax";
x1all = load('class_1','-ascii');
x2all = load('class_2','-ascii');
x3all = load('class_3','-ascii');
%% Exploritary data analysis
% Histograms
% Sepal length
if plot_histograms == true
    fig = figure;
    subplot(2,2,1);
    nbins = 30;
    samples = 1:50;
    histogram(x1all(samples,1),nbins);
    hold on;
    histogram(x2all(samples,1),nbins);
    hold on;
    histogram(x3all(samples,1),nbins);
    legend("Iris Setosa", "Iris Versicolour", "Iris Virginica",'interpreter','latex');
    title("Sepal length",'Interpreter','latex');
    xlabel("length [cm]",'Interpreter','latex');
    grid on;
    
    %Sepal width
    subplot(2,2,2);
    nbins = 30;
    histogram(x1all(samples,2),nbins);
    hold on;
    histogram(x2all(samples,2),nbins);
    hold on;
    histogram(x3all(samples,2),nbins);
    legend("Iris Setosa", "Iris Versicolour", "Iris Virginica",'interpreter','latex');
    title("Sepal width",'Interpreter','latex');
    xlabel("width [cm]",'Interpreter','latex');
    grid on;
    
    %Petal length
    subplot(2,2,3);
    nbins = 30;
    histogram(x1all(samples,3),nbins);
    hold on;
    histogram(x2all(samples,3),nbins);
    hold on;
    histogram(x3all(samples,3),nbins);
    legend("Iris Setosa", "Iris Versicolour", "Iris Virginica",'interpreter','latex');
    title("Petal length",'Interpreter','latex');
    xlabel("length [cm]",'Interpreter','latex');
    grid on;
    
    %Petal width
    subplot(2,2,4);
    nbins = 30;
    histogram(x1all(samples,4),nbins);
    hold on;
    histogram(x2all(samples,4),nbins);
    hold on;
    histogram(x3all(samples,4),nbins);
    legend("Iris Setosa", "Iris Versicolour", "Iris Virginica",'interpreter','latex');
    title("Petal width",'Interpreter','latex');
    xlabel("width [cm]",'Interpreter','latex');
    grid on;
end

%% Remove sepal length
if remove_sepal_length == true
    x1all(:,1) = [];
    x2all(:,1) = [];
    x3all(:,1) = [];
end
%% Remove sepal width
if remove_sepal_width == true
    x1all(:,1) = [];
    x2all(:,1) = [];
    x3all(:,1) = [];
end
%% Remove petal lenght
if remove_petal_length == true
    x1all(:,1) = [];
    x2all(:,1) = [];
    x3all(:,1) = [];
end
%% Data pre-processing
% Input normalization
Max = max([x1all;x2all;x3all]);
%max finds the biggest element in each colum(i.e. feature)
Min = min([x1all;x2all;x3all]);
%min finds the smallest element in each colum(i.e. feature)
x1all = (x1all-Min)./(Max-Min);
x2all = (x2all-Min)./(Max-Min);
x3all = (x3all-Min)./(Max-Min);

% Homogeneus form


x1train = [ones(length(trainSamples),1), x1all(trainSamples,:)];
x2train = [ones(length(trainSamples),1), x2all(trainSamples,:)];
x3train = [ones(length(trainSamples),1), x3all(trainSamples,:)];

x1test = [ones(length(testSamples),1), x1all(testSamples,:)];
x2test = [ones(length(testSamples),1), x2all(testSamples,:)];
x3test = [ones(length(testSamples),1), x3all(testSamples,:)];


% Merge training data for ease of training
xtrain = [x1train; x2train; x3train]';
% One-hot encoding
encodingClass1 = [1; 0; 0];
encodingClass2 = [0; 1; 0];
encodingClass3 = [0; 0; 1];
% Create correct output values for training
ytrain = [repmat(encodingClass1,1,length(trainSamples))...
    , repmat(encodingClass2,1,length(trainSamples))...
    , repmat(encodingClass3,1,length(trainSamples))];

% Group test data in cells for easy evaluation
numClasses = 3;


xtest = cell(1,numClasses);
xtest{1} = x1test';
xtest{2} = x2test';
xtest{3} = x3test';

%clearvars -except xtrain ytrain xtest numClasses
numFeatures = height(xtrain)-1;


%% Training

% Hyperparameters
numEpochs = 1500;
alpha = 0.3;
%Random initial values
W = randn(numClasses, numFeatures+1);% +1 for bias term
Errors = zeros(1,numEpochs);
tic;
for j = 1:numEpochs
    gradW = zeros(numClasses, numFeatures+1);
    E = 0;
    for i = 1:length(xtrain)
        x = xtrain(:,i);
        y = ytrain(:,i);
        if training_mode == "sigmoid"
            g = sig(W*x);
            gradW = gradW + (((g-y).*g).*(1-g))*x';
            E = E + (g-y)'*(g-y);
        elseif training_mode == "softmax"
            g = softmax(W*x);
            gradW = gradW -(softmaxGradient(W*x)*(y./g))*x';
            E = E - log(g)'*y;
        end
    end
    Errors(j) = E;
    W = W - alpha*gradW;
    fprintf("Epoch: %d,Learning rate: %f, Error: %f\n", j, alpha, E);
end
t = toc;
fprintf("Training took %.2f seconds",t);
fig = figure;
plot(Errors);
grid on;
xlabel("Epoch",'Interpreter','latex');
ylabel("Loss function value",'Interpreter','latex');
title("Loss function as a function of epoch number, for sigmoid activation function",'Interpreter','latex');
%exportgraphics(fig,"L2error_softmax.pdf")

%% Evaluation
confMatrixtest = zeros(numClasses,numClasses);
confMatrixtrain = zeros(numClasses,numClasses);
for j = 1:length(xtrain)
    x = xtrain(:,j);
    y = find(ytrain(:,j)==max(ytrain(:,j)));
    g = W*x;
    class = find(g == max(g));
    % Count classifications on training dataset
    confMatrixtrain(y,class) = confMatrixtrain(y,class) + 1;
end
errorRateTrain = (sum(confMatrixtrain,'all')-sum(diag(confMatrixtrain)))/length(xtrain);

for i = 1:numClasses
    for j = 1:length(xtest{i})
        x = xtest{i}(:,j);
        g = W*x;
        class = find(g == max(g));
        % Count classifications on training dataset
        
        confMatrixtest(i,class) = confMatrixtest(i,class) + 1;
    end
end
errorRateTest = (sum(confMatrixtest,'all')-sum(diag(confMatrixtest)))/(3*length(xtest{1}));

%confMatrixtest = confMatrixtest/length(xtest{1})*100;
%confMatrixtrain = confMatrixtrain/(length(xtrain)/numClasses)*100;

disp("Confusion matrix training set:");
disp(confMatrixtrain);
fprintf("Error rate training set: %.2f \n",errorRateTrain);
disp("Confusion matrix test set:");
disp(confMatrixtest);
fprintf("Error rate test set: %.2f\n",errorRateTest);




function sig = sig(x)
    sig = 1./(1+exp(-x));
end

function soft = softmax(x)
    soft = exp(x)/sum(exp(x));
end


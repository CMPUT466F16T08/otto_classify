close all;
clear all;

load('../train_set.mat');
load('../test_set.mat');

%mdl = fitcecoc(Xtrain,Ytrain,'FitPosterior',1);  
load('svmmodel_v0.mat');
[label,~,~,prob] = predict(mdl,Xtest);


index=sub2ind(size(prob),(1:size(Ytest,1))',Ytest);

true_prob=prob(index);
%true_prob(true_prob==0)=1e-15;

logloss=-sum(log(true_prob))/size(Ytest,1);
% 1e-6 0.615749172276437 
% 1e-15 0.615749172276437
% 0 0.615749172276437
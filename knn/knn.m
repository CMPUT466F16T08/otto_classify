close all;
clear all;

run('../vlfeat-0.9.20/toolbox/vl_setup')

load('../train_set.mat');
load('../test_set.mat');

Xtrain=Xtrain';
Xtest=Xtest';
Ytrain=Ytrain';
Ytest=Ytest';

kdtree = vl_kdtreebuild(Xtrain) ;
[index, distance] = vl_kdtreequery(kdtree, Xtrain, Xtest);
Ypred=Ytrain(index);
sum(Ypred-Ytest)
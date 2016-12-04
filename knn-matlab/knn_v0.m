close all;
clear all;

load('../train_set.mat');
load('../test_set.mat');

mdl = fitcknn(Xtrain,Ytrain,'NumNeighbors',5,'Crossval','on');  

[label,score,cost] = predict(mdl,Xtest);


index=sub2ind(size(score),(1:size(Ytest,1))',Ytest);

true_prob=score(index);
true_prob(true_prob==0)=1e-6;

logloss=-sum(log(true_prob))/size(Ytest,1);
%logloss 1.133133720104564
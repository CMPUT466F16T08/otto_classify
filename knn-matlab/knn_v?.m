close all;
clear all;

load('../train_set.mat');
load('../test_set.mat');
totalnn=20;
ces=zeros(totalnn,1);
for i=1:totalnn
    mdl = fitcknn(Xtrain,Ytrain,'NumNeighbors',i,'Crossval','on','KFold',5,'Standardize',1); 
    classError = kfoldLoss(mdl);
    ces(i,1)=classError;
end

[~,optnn]=min(ces);
mdl = fitcknn(Xtrain,Ytrain,'NumNeighbors',optnn); 

[label,score,cost] = predict(mdl,Xtest);


index=sub2ind(size(score),(1:size(Ytest,1))',Ytest);

true_prob=score(index);
true_prob(true_prob==0)=1e-6;

logloss=-sum(log(true_prob))/size(Ytest,1);
%logloss 0.847753484767668

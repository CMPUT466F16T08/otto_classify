close all;
clear all;

load('../train_set.mat');
load('../test_set.mat');
totalnn=20;
ces=zeros(totalnn,1);
for i=1:totalnn
    mdl = fitcknn(Xtrain,Ytrain,'NumNeighbors',i,'Crossval','on','KFold',10,'Standardize',1); 
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
%[0.245060805624316;0.258595612298815;0.240495333522173;0.236273281887891;0.233909741020852;0.234010747040812;0.232536059149411;0.232374449517477;0.230212920690356;0.231465395337847;0.231990626641633;0.232778473597313;0.232414851925461;0.233061290453198;0.233485515737026;0.234030948244804;0.233929942224845;0.234535978344599;0.235303624096286;0.235970263828016]
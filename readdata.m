close all;
clear all;

%fid = fopen('train.csv');
fidtrain = fopen('train_set.csv','r');
fidtest = fopen('test_set.csv','r');
%out = textscan(fid,[repmat('%f',1,94),'Class_%f'],'delimiter',',','HeaderLines',1);
outtrain = textscan(fidtrain,repmat('%f',1,95),'delimiter',',','HeaderLines',1);
outtest = textscan(fidtest,repmat('%f',1,95),'delimiter',',','HeaderLines',1);
fclose(fidtrain);
fclose(fidtest);

mtrain=cell2mat(outtrain);
mtest=cell2mat(outtest);
Train=mtrain(:,2:end);
Test=mtest(:,2:end);
Xtrain=Train(:,1:end-1);
Ytrain =Train(:,end);
Ytrain =Ytrain+1;
Xtest=Test(:,1:end-1);
Ytest=Test(:,end);
Ytest=Ytest+1;
%save('train.mat','M');
save('train_set.mat','Xtrain','Ytrain');
save('test_set.mat','Xtest','Ytest');
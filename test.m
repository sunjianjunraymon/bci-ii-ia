clear;clc;
tic
load('zongshuju.mat');
load('ceshilabel.mat');

wn=[8/128 30/128];  %8~30Hz Bandpass filter
b = fir1(50,wn,'DC-0');

traindata=xunliandata;
trainlabel=xunlianlabel;
testdata=ceshidata;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbChannels = size(traindata,2);      % 通道
nbTrials = size(traindata,3);        % 实验次数
classLabels=[1 2];
nbClasses = length(classLabels);        % 类别

covMatrices = cell(nbClasses,1);
trialCov = zeros(nbChannels,nbChannels,nbTrials);
for t=1:nbTrials
    E = traindata(:,:,t)';                       %note the transpose
    EE = E * E';
    trialCov(:,:,t)=EE./trace(EE);
end

for c=1:nbClasses      
   covMatrices{c} = mean(trialCov(:,:,trainlabel == classLabels(c)),3);  
end
covTotal = covMatrices{1} + covMatrices{2};

[Ut,Dt] = eig(covTotal); 
eigenvalues = diag(Dt);
[eigenvalues,egIndex] = sort(eigenvalues, 'descend');
Ut = Ut(:,egIndex);
P = diag(sqrt(1./eigenvalues)) * Ut';
transformedCov1 =  P * covMatrices{1} * P';
[U1,D1] = eig(transformedCov1);
eigenvalues = diag(D1);
[eigenvalues,egIndex] = sort(eigenvalues, 'descend');
U1 = U1(:, egIndex);
CSPMatrix = U1' * P;

%下面是提取特征%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
nbFilterPairs=3;
nbTrials = size(traindata,3);
features = zeros(nbTrials, 2*nbFilterPairs+1);
Filter = CSPMatrix([1:nbFilterPairs (end-nbFilterPairs+1):end],:);
%extracting the CSP features from each trial
for t=1:nbTrials    
    projectedTrial = Filter * traindata(:,:,t)';  
    variances = var(projectedTrial,0,2);
    for f=1:length(variances)
        features(t,f)=variances(f)/sum(variances);
    end
end
features(:,end)=trainlabel;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[features(:,1:end-1),da,xiao]=suoyouliezuidazuixiao(features(:,1:end-1));
for i=1:size(features,2)-1
  % seed(i).real=ningjucengcijulei(features(:,i),0.005);
  % seed(i).real=ningjucengcijulei(features(:,i),0.006);
 %  seed(i).real=ningjucengcijulei(features(:,i),0.002);
  %   seed(i).real=ningjucengcijulei(features(:,i),0.007);
 % seed(i).real=ningjucengcijulei(features(:,i),0.01);   
 % seed(i).real=ningjucengcijulei(features(:,i),0.003);
  %   seed(i).real=ningjucengcijulei(features(:,i),0.001);
     seed(i).real=ningjucengcijulei(features(:,i),0.004);
    %  seed(i).real=ningjucengcijulei(features(:,i),0.009);
  %   seed(i).real=ningjucengcijulei(features(:,i),0.008);
    seed(i).imag=i;
    toc;
end

seedtemp=seed;
%以下是把种子行数小于5的删除
for i=1:size(features,2)-1
     temp=seedtemp(i).real; 
     for j=1:size(temp,1)
          if sum(temp(j,:)>0)<4
              seedtemp(i).real(j,:)=zeros(1,size(temp,2));
           end
     end
     seedtemp(i).real(all(seedtemp(i).real==0,2),:) = []; 
end

deta=0.9;  %deta 0.1-0.25
i=1;

for m=1:size(seedtemp,2) %319
     for n=1:size(seedtemp(m).real,1) %行数
          temp2=seedtemp(m).real(n,:); %
         [score expand]=MES(temp2,deta,seedtemp,features(:,1:end-1));
         rule(i).real=expand
         rule(i).imag=score;
          toc;
          i=i+1
     end
end 

pos=[];Rule=[];v=[];
n=0;
for i=1:size(rule,2)
    if ~isempty(rule(i).real)
        temp=rule(i).real;
             for j=1:size(temp,2)
                 if sum(temp(:,j))>0
                   v(j)=mean(features(temp(:,j)));   
                   pos=temp(:,j);                                                                                    
                 end
             end
             if sum(pos)>0
                  n=n+1;                                                                                       
                  if mean(trainlabel(pos))>1.65
                      label(n)=2;
                      weizhi(n)=i;
                  elseif mean(trainlabel(pos))<1.35
                      label(n)=1;
                       weizhi(n)=i;
                  else
                      label(n)=0;
                      weizhi(n)=i;
                  end 
             end

    end
    v(length(v)+1:2*nbFilterPairs)=0;
    Rule(i).real=v;
    v=[];
end
RR=[];
RL=[];
for i=1:n
   if label(i)==2
      FPR=[Rule(weizhi(i)).real 2];
      RR=[RR; FPR];
   elseif label(i)==1
      FPL=[Rule(weizhi(i)).real 1];
      RL=[RL; FPL];
   end
end

df=[];
for i=1:size(features,1)
 df=[df fuinf(RL,RR,features(i,1:end-1))];   
 toc
end

toc

P=df;
T=trainlabel';
[yuzhi]=pso(P,T);

toc

nbtestTrials = size(testdata,3);
tfeatures = zeros(nbtestTrials, 2*nbFilterPairs);
Filter = CSPMatrix([1:nbFilterPairs (end-nbFilterPairs+1):end],:);
for t=1:nbtestTrials    
    projectedTrial = Filter * testdata(:,:,t)';  
    variances = var(projectedTrial,0,2);
    for f=1:length(variances)
        tfeatures(t,f)=variances(f)/sum(variances);
    end
end
tfeatures=lyqmdxgyh(tfeatures,da,xiao);

dft=[];
for i=1:size(tfeatures,1)
 dft=[dft fuinf(RL,RR,tfeatures(i,1:end))];   
 toc
end

for i=1:numel(dft)
    if dft(i)>yuzhi
        pl(i)=2;
    else
        pl(i)=1;
    end
end

cha=testlabel-pl;
n=0;
for i=1:numel(cha)
   if cha(i)==0
       n=n+1;
   end
end

acc=n/numel(testlabel)













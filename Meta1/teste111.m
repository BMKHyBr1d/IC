close all; clear; clc 

classNames = 0:9;
Teste=load('notMNIST_small.mat')
% Treino=load('notMNIST_large.mat')
% 
% 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Treino%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %  Treino.images = reshape(Treino.images, size(Treino.images, 1) * size(Treino.images, 2), size(Treino.images, 3));
% % % Convert to double and rescale to [0,1]
% % Treino.images = double(Treino.images) / 255;
% 
%   conjuntoDeTreino_x = Treino.images;
% conjuntoDeTreino_y_temp = Treino.labels;
% conjuntoDeTreino_y=squeeze(onehotencode(conjuntoDeTreino_y_temp,10,'ClassNames',classNames));
% % disp(size(conjuntoDeTreino_x));
% % disp(size(conjuntoDeTreino_y));
% % 
% %  figure; colormap(gray)
% %  for i=1:25
% %      subplot(5,5,i)
% %      digit = reshape(Treino.images(:,i),[28 28]);
% %      imagesc(digit)
% %       disp(Treino.labels(i));
% %  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Teste%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Teste.images = reshape(Teste.images, size(Teste.images, 1) * size(Teste.images, 2), size(Teste.images, 3));
% Convert to double and rescale to [0,1]
Teste.images = double(Teste.images) / 255;

 conjuntoDeTeste_x = Teste.images;
conjuntoDeTeste_y_temp = Teste.labels;
conjuntoDeTeste_y=squeeze(onehotencode(conjuntoDeTeste_y_temp,10,'ClassNames',classNames));
disp(size(conjuntoDeTeste_x));
disp(size(conjuntoDeTeste_y));


 figure; colormap(gray)
 for i=1:25
     subplot(5,5,i)
     digit = reshape(Teste.images(:,i),[28 28]);
     imagesc(digit)
      disp(Teste.labels(i));
 end

%%%%%%%%%%%%%%%%%%%plot%%%%%%%%%%%%%%%%%%%



MT = conjuntoDeTeste_y';
P=conjuntoDeTeste_x;

net = patternnet(10);
net.divideFcn = 'dividerand';
%net.trainFcn= 'traingd';
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 20/100;
net.trainParam.lr=0.01;
net.trainParam.epochs=1000;
net = train(net,conjuntoDeTeste_x,conjuntoDeTeste_y');

y = round(net(P));
plotconfusion(y,MT)
%%%%%%%%%%%%%%%%%%Resultados%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[c,cm,ind,per] = confusion(MT,y);
TotalP=0;
TotalA=0;
TotalR=0;
TotalE=0;
for j=1 :10
    FN=0;
    FP=0;
    for i=1 : 10
        if(j~=i)
            FP= FP+(cm(j,i));
            FN=FN+(cm(i,j));
        end
    end
    TP=cm(j,j);
    TN=sum(sum(cm(:,:)))-(TP+FP+FN);
    C = sprintf('Resultados para a classe %d',j);
      disp(C)
      Precision = TP/(TP+FP);
      X = sprintf('%f Precision\n',Precision);
      disp(X)
    TotalP = TotalP + Precision;
      Recall = TP/ (TP+FN);
       X = sprintf('%f Recall\n',Recall);
      disp(X)
    TotalR = TotalR + Recall;
      Accuracy = (TP+TN) / (TP+TN+FN+FP);
        X = sprintf('%f Accuracy\n',Accuracy);
        disp(X)
    TotalA = TotalA + Accuracy;
     Espec = TN/(FP+TN);
          X = sprintf('%f Especificidade\n',Espec);
        disp(X)
    TotalE = TotalE + Espec;
    
    TPR= TP/(TP+FN);
    FPR = FP/(FP+TN);
    A = [0;TPR;1];
    B = [0;FPR;1];
    AUC = trapz(B,A);
    X = sprintf('%f  Valor AUC',AUC);
    disp(X)
end

disp('-----Media-----');
fprintf('%f Precision\n',TotalP);
fprintf('%f Recall\n',TotalR);
fprintf('%f Accuracy\n',TotalA);
fprintf('%f Especificidade\n',TotalE);


 disp('-----Fmesure-----\n');
Fmesure = ((TotalP*TotalR/(TotalP + TotalR))*2)/10;
fprintf('%f\n',Fmesure);







 
% clear all
% close all
% 
% R=[ 5 3 0 1; 
%     4 0 0 1; 
%     1 1 0 5; 
%     1 0 0 4; 
%     0 1 5 4];
% [nRow, nCol]=size(R);
% K=2;
% P=rand(nRow, K);
% Q=rand(K, nCol);
% 
% steps=5000;
% alpha=0.0002;
% beta=0.02;
% 
% 
% [nP, nQ]=matrix_factorization(R,P,Q,K, steps, alpha, beta);
% [nP*nQ]
% [R]



R=[ 4 3 0 1 2; 
    5 0 0 1 0; 
    1 2 1 5 4; 
    1 0 0 4 0; 
    0 1 5 4 0;
    5 5 0 0 1];
[nRow, nCol]=size(R)
K=3;
P=rand(nRow, K);
Q=rand(K, nCol);


steps=10000;
alpha=0.0002;
beta=0.02;


[nP, nQ]=matrix_factorization(R,P,Q,K, steps, alpha, beta)
[nP*nQ]
[R]

function [nP, nQ]=matrix_factorization(R,P,Q,K,steps,alpha,beta)
[m,n] = size(R);
%     R(R==0)=NaN;
% steps1 = steps;
nP = P;
nQ = Q;
while steps>0
    e = R-P*Q;
    for i = 1:m
        for j = 1:n
            if isnan(R(i,j))
                continue
            end
            nP(i,:) = P(i,:)+alpha*(2*e(i,j)*Q(:,j).'-beta*P(i,:));
            nQ(:,j) = Q(:,j)+alpha*(2*e(i,j)*P(i,:).'-beta*Q(:,j));
            P = nP;
            Q = nQ;
        end
    end
    steps = steps-1;
%     e1(:,:,steps1+1-steps) = e;
%     steps = steps-1;
end
% plot(reshape(e1(1,1,:),[1,steps1]));
% hold on;
% plot(reshape(e1(1,2,:),[1,steps1]));
% hold off;
end







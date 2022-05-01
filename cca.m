function [Wa,Wb,S,Za,Zb] = cca(Xa,Xb)
%
% function [Wa,Wb,S,Za,Zb] = cca(Xa,Xb)
% INPUTS
% Xa - Matrix 1, of size n rows (observations) and p columns (variables).
% Xb - Matrix 2, of size n rows (observations) and q columns (variables).
% OUTPUTS
% Wa - CCA weights (loadings) applied to variables of Xa. 
% Wb - CCA weights (loadings) applied to variables of Xb
% Size of Wa and Wb determined by the min(p,q)
% S - the correlation between the low dimnesional projections. 
% Za - Canonical variates of Xa of dimension n rows and min(p,q) cols
% Zb - Canonical variates of Xb of dimension n rows and min(p,q) cols
%
%
% MAIN CCA
% Nikhilesh Natraj, Neuron 2022

% % de-mean
Xa = Xa-mean(Xa);
Xb = Xb-mean(Xb);

% sample auto and cross covariance matrices
Caa = ((size(Xa,1)-1)^-1)* (Xa'*Xa);
Cbb = ((size(Xa,1)-1)^-1)* (Xb'*Xb);
Cab = ((size(Xa,1)-1)^-1)* (Xa'*Xb);

% check for rank
if rank(Caa)<size(Caa,1)
    Caa = Caa + 1e-8*eye(size(Caa));
end

if rank(Cbb)<size(Cbb,1)
    Cbb = Cbb + 1e-8*eye(size(Cbb));
end

% cholesky factorization: numerical stability
Caa12=chol(Caa);
Cbb12=chol(Cbb);

% CCA solver
X = (Caa12')\Cab/(Cbb12);
[U,S,V]=svd(X,0); % Singular values - uncovered canonical correlations
Wa = Caa12\U; % Each column of Wa gives a weighting of variables of Xa
Wb = Cbb12\V; % Each column of Wb gives a weighting of variables of Xb
S=diag(S);

% canonical variates
Za=Xa*Wa;
Zb=Xb*Wb;

end



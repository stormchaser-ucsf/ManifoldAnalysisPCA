function [Xa,W]  = sphere_data(Xa)
% function [Xa,W]  = sphere_data(Xa)


[v,d]=eig(cov(Xa));
V1 = v';
D = sqrt(inv(d));
Xa=Xa';
%W=D*V1; %whitening
W=V1'*D*V1; %sphereing
Xa=W*Xa;
Xa=Xa';


end
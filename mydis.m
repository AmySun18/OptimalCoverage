function dis=mydis(X,s)
% this function will return each point in X dimension(m*2) distance with
% point s dimension(1*2)
% dis is a column 
[r c]=size(X);
S=ones(r,1)*s;
dis=sqrt(sum((S-X).^2'))';
end
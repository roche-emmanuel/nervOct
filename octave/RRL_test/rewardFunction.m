function [ Ret, sharp ] = rewardFunction( X, miu, delta, Ft, M)
%REWARDFUNCTION Summary of this function goes here
%   Detailed explanation goes here
    T = length(Ft)-1;

    % Ft(1:10)
    % X(M:M+9)

    % Ret = miu * (Ft(1:T) .* X(M+1:M+T) - delta * abs(Ft(2:end)-Ft(1:T)));
    Ret = miu * (Ft(1:T) .* X(M:M+T-1) - delta * abs(Ft(2:end)-Ft(1:T)));
   
   	% n = size(Ret,1)

    sharp = sharpRatio(Ret);
end


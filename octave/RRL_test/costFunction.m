function [ J, grad ] = costFunction( Xn, X, theta)
%COSTFUNCTION Summary of this function goes here
%   Detailed explanation goes here
    miu = 1;
    delta = 0.001;
        
    M = length(theta) - 2;
    T = length(X) - M;

    Ft = updateFt(Xn, theta, T);     
    
    %Ft(1:10)

    [Ret, sharp] = rewardFunction(X, miu, delta, Ft, M);
    J = sharp * -1;
    Ret(1:10)

    dFt = zeros(M+2,T+1);
    for i = 2:T+1,
        xt = [1; Xn(i-1:i+M-2); Ft(i-1)];
        dFt(:,i) = (1 - tanh(theta' * xt) ^ 2) * (xt + theta(M+2,1)*dFt(:,i-1));
    end
    
    
    dRtFt = -1 * miu * delta * sign(Ft(2:end)-Ft(1:T));
    %dRtFtt = miu * X(M+1:T+M) + miu * delta * sign(Ft(2:end)-Ft(1:T));
    dRtFtt = miu * X(M:M+T-1) + miu * delta * sign(Ft(2:end)-Ft(1:T));
    A = sum(Ret) / T;
    B = sum(Ret.*Ret) / T;

    dsda = dSdA(A,B);
    dsdb = dSdB(A,B);

    prefix = repmat(dsda, T, 1)/T + dsdb*2*Ret/T;
    %prefix = repmat((1/(- A^2 + B)^(1/2) + A^2/(B - A^2)^(3/2))/M, M, 1) + (-A/(2*(B - A^2)^(3/2))) * 2 * Ret / M
    %prefix = ones(M,1)
    
    grad = sum(repmat(prefix', M+2, 1) .* (repmat(dRtFt', M+2, 1) .* dFt(:,2:end) + repmat(dRtFtt', M+2, 1) .* dFt(:,1:T)), 2);
    grad = grad * -1;

end

function val = dSdA(A,B)
    val = B / (B - A*A)^(3/2);
end

function val = dSdB(A,B)
    val = -0.5 * A / (B - A*A)^(3/2);
end

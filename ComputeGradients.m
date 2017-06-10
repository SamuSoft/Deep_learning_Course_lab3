function [grad_W, grad_b, g] = ComputeGradients(W, Y_Data,G, P, S, GDparams)
    lambda = GDparams{3};
    grad_W = 0;
    grad_b = 0;
    g = zeros(size(S,1),size(Y_Data,2));

    for i = 1:size(Y_Data,2)
        grad_b = grad_b + G(:,i);
        grad_W = grad_W + (G(:,i)*P(:,i)');

        next_layers_G = G(:,i)' * W;
        g(:,i) = (next_layers_G*diag((S(:,i)>0)))';
    end
    grad_b = grad_b./size(Y_Data,2);
    grad_W = grad_W./size(Y_Data,2) + 2*lambda*W;
end

function J = ComputeCost(X, Y, W_layers, b_layers, lambda)
    P = EvaluateClassifier(X,W_layers,b_layers);
    P = P{3};
    sum_p = 0;
    for i = 1:size(Y,2)
        sum_p = sum_p + l_cross(Y(:,i),P(:,i));
    end
    J = 1/size(X,2)*sum_p + lambda*(sum(sum(W_layers{1}.^2))+sum(sum(W_layers{2}.^2)));
end

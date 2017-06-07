function acc  = ComputeAccuracy(X, y, W, b, Error_Type)
    labeled_data = ones(size(X,2));
    P = ones(size(X,2));
    for i = 1:size(X,2)
        s1 = W{1}*double(X(:,i)) +b{1};
        h = max(0,s1);
        s = W{2}*h + b{2};
        p = SOFTMAX(s);
        [~,I] = max(p);
        P(i) = p(I);
        labeled_data(i)= I;
    end
    val = 0;
    for i = 1:size(y,1)
        if labeled_data(i) == y(i)
            if strcmp('RMSE',Error_Type)
                val = val + (P(i) - 1)^2;
            elseif strcmp('MSE',Error_Type)
                val = val + (P(i) - 1)^2;
            else
                val = val+1;
            end
        end
    end
    %disp(val);
    if strcmp(Error_Type, 'RMSE')
        acc = sqrt(val/size(y,1));
    else
        acc = val/size(y,1);
    end

end

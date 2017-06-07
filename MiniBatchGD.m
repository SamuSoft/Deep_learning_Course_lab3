function [new_layers] = MiniBatchGD(X, Y, GDparams, layers, lambda)
    % unpack layers
    W_layers = layers{1};
    b_layers = layers{2};
    V_list = layers{3};

      % Momentum for the different layer parameters
        % V_list{1}{1} for W_layers{1}
        % V_list{1}{2} for W_layers{2}
        % V_list{2}{1} for b_layers{1}
        % V_list{2}{2} for b_layers{2}

    % set variables
    N = size(X,2);

    % unpack parameters
    n_batch = GDparams{1};
    eta = GDparams{2};
    n_epochs = GDparams{3};
    rho = GDparams{4};
%     GDparams = {n_batch, eta, n_epochs, rho}

%     for i = 1:n_epochs
    for j = 1:N/n_batch
%             Sets the parts of the dataset for this batch
        Xtrain = X(:,((j-1)*n_batch + 1):(j*n_batch));
        Ytrain = Y(:,((j-1)*n_batch + 1):(j*n_batch));
        P = EvaluateClassifier(Xtrain, W_layers, b_layers);
        [grad_W, grad_b] = ComputeGradients(Xtrain, Ytrain, P, W_layers, b_layers, lambda);

    %     Update Layer 2
    % ------------------------------------------------
        % Update momentum vectors
        V_list{1}{2} = (rho.*V_list{1}{2}) + eta*grad_W{2};
        V_list{2}{2} = (rho.*V_list{2}{2}) + eta*grad_b{2};
        % Update second layer weights
        W_layers{2} = W_layers{2} - V_list{1}{2};
        b_layers{2} = b_layers{2} - V_list{2}{2};
    % ------------------------------------------------

    %     Update Layer 1
    % ------------------------------------------------
        % Update momentum vectors
        V_list{1}{1} = (rho.*V_list{1}{1}) + eta*grad_W{1};
        V_list{2}{1} = (rho.*V_list{2}{1}) + eta*grad_b{1};
        % Update first layer weights
        W_layers{1} = W_layers{1} - V_list{1}{1};
        b_layers{1} = b_layers{1} - V_list{2}{1};
    % ------------------------------------------------
    end
    new_layers = {W_layers, b_layers, V_list};
end

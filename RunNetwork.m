function [Data, Model] = RunNetwork(Train_Data, Test_Data, GDparams)

  % Extract Data from GDparams
  n_epochs              = GDparams{3};
  hidden_layer_nodes_1  = GDparams{7};
  standard_deviation    = GDparams{5};
  lambda                = GDparams{6};
  % Extract data from training datasets
  Xtrain = Train_Data{1};
  Ytrain = Train_Data{2};
  ytrain = Train_Data{3};
  % Extract data from test datasets
  Xtest = Test_Data{1};
  Ytest = Test_Data{2};
  ytest = Test_Data{3};

  disp('Initializing parameters')
  layers = init_param(standard_deviation, hidden_layer_nodes_1, size(Xtest,1), size(Ytest,1));

  acc = zeros(2,n_epochs);
  loss = zeros(2,n_epochs);
  disp('Starting Mini Batch Gradient Decent')
  fprintf('Epoch = 0');

  for i = 1:n_epochs

    % layers = {W_layers, b_layers, V_list};
    [new_layers] = MiniBatchGD(Xtrain, Ytrain, GDparams, layers, lambda);
    layers = new_layers;

    % Decay the moment vectors.
    for k = 1:size(layers{3},2)
      for j = 1:size(layers{3}{k},2)
        layers{3}{k}{j} = layers{3}{k}{j};
      end
    end

    %   Prints out which epoch you are in
    % ------------------------------------------------
    if i < 11
      fprintf(' \b\b%d', i);
    elseif i < 101
      fprintf(' \b\b\b%d', i);
    else
      fprintf(' \b\b\b\b%d', i);
    end
    % ------------------------------------------------
    W_layers = layers{1};
    b_layers = layers{2};
    loss(1,i) = ComputeCost(Xtrain, Ytrain, W_layers, b_layers, lambda);
    loss(2,i) = ComputeCost(Xtest, Ytest, W_layers, b_layers, lambda);
    acc(1,i) = ComputeAccuracy(Xtrain, ytrain, W_layers, b_layers, 'RMSE');
    acc(2,i) = ComputeAccuracy(Xtest, ytest, W_layers, b_layers, 'RMSE');
  end

  fprintf('\n');

  Data = {acc, loss};
  Model = {layers{1}, layers{2}};

end

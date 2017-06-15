classdef NeuralNetwork < matlab.mixin.SetGet

    properties
      layer_list;
      network_size;
    end
    properties (GetAccess=private)
      BatchNormalize_Var = false;
    end

    methods
      function obj = NeuralNetwork()
        obj.network_size = 0;
      end
      function add(obj, layer)
        if isa(layer,'NeuralLayer')
          % obj.layer_list{obj.network_size} = layer;
          set(obj,'layer_list', [obj.layer_list,layer]);
          set(obj,'network_size', obj.network_size+1);
        else
          error('Not a neural layer')
        end
      end
      function train(obj, train_data, GDparams, varargin)
          n_Epochs = GDparams{4};
          batch_size = GDparams{5};
          eta = GDparams{2};
          lambda= GDparams{3};
          X_Data = train_data{1};
          Y_Data = train_data{2};
          disp('Starting Mini Batch Gradient Decent');
          fprintf('Epoch = 0');
          if any(strcmpi('BatchNormalize',varargin)) == 1
              set(obj,'BatchNormalize_Var',true);
          end
          for i = 1:n_Epochs
              train_datasets(obj, X_Data, Y_Data, i, GDparams, varargin{:});
              rho = .9*GDparams{1};
              GDparams = {rho, eta, lambda, n_Epochs, batch_size};
          end
          fprintf('\n');
      end
      function Loss = trainWithLoss(obj, train_data, test_data, GDparams,varargin)
          n_Epochs = GDparams{4};
          batch_size = GDparams{5};
          Loss = zeros(1,n_Epochs);
          X_Data = train_data{1};
          Y_Data = train_data{2};
          Test_X = test_data{1};
          Test_y = test_data{2};
          eta = GDparams{2};
          lambda= GDparams{3};
          % disp(any(strcmpi('BatchNormalize',varargin)))
          if any(strcmpi('BatchNormalize',varargin)) == 1
              set(obj,'BatchNormalize_Var',true);
              disp('Initiating Batch Normalization');
          end

          disp('Starting Mini Batch Gradient Decent');
          fprintf('Epoch = 0');
          for i = 1:n_Epochs
              train_datasets(obj, X_Data, Y_Data, i, GDparams, varargin{:});
              rho = .9*GDparams{1};
              GDparams = {rho, eta, lambda, n_Epochs, batch_size};
              Loss(1,i) = obj.ComputeCost(Test_X, Test_y, GDparams{3});
          end
          fprintf('\n');
      end
      function [S,P] = EvaluateClassifier(obj, X)
          Layers = obj.layer_list;
          for i = 1:(obj.network_size)
            P{i} = zeros(Layers(i).node_number,size(X,2));
            S{i} = zeros(Layers(i).node_number,size(X,2));
          end
          if obj.network_size == 0
            error('No Layers!')
          elseif obj.network_size == 1
            for i = 1:size(X,2)
               s = Layers(1).eval(X(:,i));
               S{1}(:,i) = s;
               h = Layers(1).activation(s);
               P{1}(:,i) = h;

            end
          else
            for i = 1:size(X,2)
               s = Layers(1).eval(X(:,i));
               if obj.BatchNormalize_Var
                 s = Layers(1).BatchNormalize(s);
               end
               S{1}(:,i) = s;
               h = Layers(1).activation(s);
               P{1}(:,i) = h;

            end
            for j = 2:(obj.network_size - 1)
              for i = 1:size(X,2)
                 s = Layers(j).eval(P{j-1}(:,i));
                 if obj.BatchNormalize_Var
                   s = Layers(j).BatchNormalize(s);
                 end
                 S{j}(:,i) = s;
                 h = Layers(j).activation(s);
                 P{j}(:,i) = h;
               end
            end
            for i = 1:size(X,2)
              s = Layers(obj.network_size).eval(P{obj.network_size-1}(:,i));
              S{obj.network_size}(:,i) = s;
              h = Layers(obj.network_size).activation(s);
              P{obj.network_size}(:,i) = h;
            end
          end
      end
      function J = ComputeCost(obj, X, Y, lambda)
          [~,P] = obj.EvaluateClassifier(X);
          P = P{obj.network_size};
          sum_p = 0;
          for i = 1:size(Y,2)
              sum_p = sum_p - log(Y(:,i)'*P(:,i));
          end
          sum_of_weights = 0;
          for i = 1:obj.network_size
              sum_of_weights = sum_of_weights + (sum(obj.layer_list(i).WeightSum^2));
          end

          J = 1/size(X,2)*sum_p + lambda*sum_of_weights;
      end
      function GradientDecent(obj, X_Data, Y_Data, GDparams)
          [S,P] = obj.EvaluateClassifier(X_Data);

          G = P{1,obj.network_size} - Y_Data;

          % if obj.BatchNormalize_Var
          %   for i = obj.network_size:-1:2
          %      G = obj.layer_list(i).BatchNormBackPass(Y_Data,G, P{i-1}, S{i-1}, GDparams);
          %      G = obj.layer_list(i).BatchNormBackPass(Y_Data,G, P{i-1}, S{i-1}, GDparams);
          %   end
          %   obj.layer_list(1).BackPass(Y_Data,G, double(X_Data), X_Data, GDparams);
          % else
          for i = obj.network_size:-1:2
             G = obj.layer_list(i).BackPass(Y_Data,G, P{i-1}, S{i-1}, GDparams);

          end
          obj.layer_list(1).BackPass(Y_Data,G, double(X_Data), X_Data, GDparams);

      end
      function train_datasets(obj, X_Data, Y_Data, epoch, GDparams, varargin)
        batch_Size = GDparams{5};
        for j = 1:1:(size(X_Data,2)/batch_Size)
          X_Batch = X_Data(:,(((j-1)*batch_Size+1):j*batch_Size));
          Y_Batch = Y_Data(:,(((j-1)*batch_Size+1):j*batch_Size));
            if obj.BatchNormalize_Var
                for i = 1:(obj.network_size)
                    X_Batch = obj.layer_list(i).SetBatchNormalization(X_Batch, varargin{:});
                end
            end
            obj.GradientDecent(X_Batch, Y_Batch, GDparams);
        end
        if epoch < 11
            fprintf(' \b\b%d', epoch);
        elseif epoch < 101
            fprintf(' \b\b\b%d', epoch);
        else
            fprintf(' \b\b\b\b%d', epoch);
        end
      end
      function SetBatchNormalization(obj, Data, varargin)
          set(obj, 'BatchNormalize_Var', true);
          if size(varargin) > 1
              if any(strcmpi('my', varargin));
                  str_index = find('my', lower(varargin));
                  set(obj, 'Train_Data_My', varargin{str_index+1});
              end
              if any(strcmpi('v', varargin));
                  str_index = find('v', lower(varargin));
                  set(obj, 'Train_Data_v', varargin{str_index+1});
              end
          else
              my = (1/size(s,2)).*sum(Data,2);
              set(obj, 'Train_Data_My', my);

              for j = 1:size(s,1)
                  sum_s = 0;
                  for i = 1:size(s,2)
                      sum_s = sum_s + ((s(j,i) - my(j))^2);
                  end
                  v(j) = (1/size(s,2))*sum_s;
              end
              set(obj, 'Train_Data_v', v);
          end
      end
      function [W_layers, b_layers] = getWb(obj)
        for i = 1:obj.network_size
          W_layers{i} = obj.layer_list(i).getW();
          b_layers{i} = obj.layer_list(i).getb();
        end
      end
      function network = copyNetwork(obj)
        network = NeuralNetwork();
        for i = 1:obj.network_size
          network.add(obj.layer_list(i).copyLayer())
        end
      end

      % Functions only to test the numerical gradient for 2 layers
      function Loss = trainNum(obj, train_data, test_data, GDparams,varargin)
          n_Epochs = GDparams{4};
          batch_Size = GDparams{5};
          Loss = zeros(1,n_Epochs);
          X_Data = train_data{1};
          Y_Data = train_data{2};
          Test_X = test_data{1};
          Test_y = test_data{2};
          eta = GDparams{2};
          lambda= GDparams{3};
          % disp(any(strcmpi('BatchNormalize',varargin)))
          if any(strcmpi('BatchNormalize',varargin)) == 1
              set(obj,'BatchNormalize_Var',true);
              disp('Initiating Batch Normalization');
              X = X_Data;
              for i = 1:(obj.network_size)
                  X = obj.layer_list(i).SetBatchNormalization(X, varargin{:});
              end
          end

          disp('Starting Mini Batch Gradient Decent');
          fprintf('Epoch = 0');
          for i = 1:n_Epochs
              train_datasets_num(obj, X_Data, Y_Data, i, GDparams);
              rho = .9*GDparams{1};
              GDparams = {rho, eta, lambda, n_Epochs, batch_Size};
              Loss(1,i) = obj.ComputeCost(Test_X, Test_y, GDparams{3});
          end
          fprintf('\n');
      end
      function train_datasets_num(obj, X_Data, Y_Data, epoch, GDparams)
        batch_Size = GDparams{5};
        for j = 1:1:(size(X_Data,2)/batch_Size)
            X = X_Data(:,(((j-1)*batch_Size+1):j*batch_Size));
            Y = Y_Data(:,(((j-1)*batch_Size+1):j*batch_Size));
            obj.ComputeGradsNumSlow(X, Y, GDparams);
        end
        if epoch < 11
            fprintf(' \b\b%d', epoch);
        elseif epoch < 101
            fprintf(' \b\b\b%d', epoch);
        else
            fprintf(' \b\b\b\b%d', epoch);
        end
      end
      function [grad_b, grad_W] = ComputeGradsNumSlow(obj, XX, Y, GDparams)
        rho = GDparams{1};
        eta = GDparams{2};
        lambda = GDparams{3};
        h = 0.000001;
        [S,P] = obj.EvaluateClassifier(XX);
        X = XX;%P{1,obj.network_size} - Y;
        for k = 1:obj.network_size
          W_layers{k} = obj.layer_list(k).getW();
          b_layers{k} = obj.layer_list(k).getb();
        end


        for k = obj.network_size:-1:2

            W_first{obj.network_size - k + 1} = obj.layer_list(k).getW();
            b_first{obj.network_size - k + 1} = obj.layer_list(k).getb();
            W = W_first{obj.network_size - k + 1};
            b = b_first{obj.network_size - k + 1};
            no = size(W, 1);
            d = size(X, 1);

            grad_W = zeros(size(W));
            grad_b = zeros(no, 1);
            clock


            for i=1:length(b)
                layers_try = b_layers;
                b_try = b;
                b_try(i) = b_try(i) - h;
                layers_try{k} = b_try;
                c1 = obj.ComputeCost_ForNum(X, Y,W_layers,layers_try,lambda);
                b_try = b;
                b_try(i) = b_try(i) + h;
                layers_try{k} = b_try;
                c2 = obj.ComputeCost_ForNum(X, Y,W_layers,layers_try,lambda);
                grad_b(i) = (c2-c1) / (2*h);
            end

            for i=1:numel(W)
                layers_try = W_layers;
                W_try = W;
                W_try(i) = W_try(i) - h;
                layers_try{k} = W_try;
                c1 = obj.ComputeCost_ForNum(X, Y, layers_try, b_layers, lambda);
                W_try = W;
                W_try(i) = W_try(i) + h;
                layers_try{k} = W_try;
                c2 = obj.ComputeCost_ForNum(X, Y, layers_try, b_layers, lambda);
                grad_W(i) = (c2-c1) / (2*h);
            end

            obj.layer_list(k).setW(W - eta*grad_W);
            obj.layer_list(k).setb(b - eta*grad_b);
            W_layers{k} = W - eta*grad_W;
            b_layers{k} = b - eta*grad_b;

            % for i = 1:size(X,2)
            %   next_layers_G = X(:,i)' * W;
            %   X(:,i) = (next_layers_G*diag(S{k-1}(:,i)>0))';
            % end

        end

        W_first{1} = obj.layer_list(1).getW();
        b_first{1} = obj.layer_list(1).getb();
        W = W_first{1};
        b = b_first{1};
        no = size(W, 1);
        d = size(XX, 1);

        grad_W = zeros(size(W));
        grad_b = zeros(no, 1);

        for i=1:length(b)
          layers_try = b_layers;
          b_try = b;
          b_try(i) = b_try(i) - h;
          layers_try{1} = b_try;
          c1 = obj.ComputeCost_ForNum(XX, Y,W_layers,layers_try,lambda);
          b_try = b;
          b_try(i) = b_try(i) + h;
          layers_try{1} = b_try;
          c2 = obj.ComputeCost_ForNum(XX, Y,W_layers,layers_try,lambda);
          grad_b(i) = (c2-c1) / (2*h);
        end
        obj.layer_list(1).setb(b);
        for i=1:numel(W)
          layers_try = W_layers;
          W_try = W;
          W_try(i) = W_try(i) - h;
          layers_try{1} = W_try;
          c1 = obj.ComputeCost_ForNum(XX, Y, layers_try, b_layers, lambda);
          W_try = W;
          W_try(i) = W_try(i) + h;
          layers_try{1} = W_try;
          c2 = obj.ComputeCost_ForNum(XX, Y, layers_try, b_layers, lambda);
          grad_W(i) = (c2-c1) / (2*h);
        end
        obj.layer_list(1).setW(W - eta*grad_W);
        obj.layer_list(1).setb(b - eta*grad_b);
        W_layers{1} = W - eta*grad_W;
        b_layers{1} = b - eta*grad_b;
      end
      function P = EvaluateClassifier_ForNum(obj,X, W, b)
          P{1} = zeros(50,size(X,2));
          P{2} = zeros(50,size(X,2));
          P{3} = zeros(10,size(X,2));
          for i = 1:size(X,2)
             s1 = W{1}*double(X(:,i)) +b{1};
             s1 = obj.layer_list(1).BatchNormalize(s1);
             P{1}(:,i) = s1;
             h = max(0,s1);
             P{2}(:,i) = h;
             s = W{2}*h + b{2};
             s = obj.layer_list(2).BatchNormalize(s);
             e = exp(s);
             one = ones(size(s,1),1);
             split = one'*e;
             P{3}(:,i) = e/split(1,1);
          end

      end
      function J = ComputeCost_ForNum(obj, X, Y, W_layers, b_layers, lambda)
          P = obj.EvaluateClassifier_ForNum(X,W_layers,b_layers);
          P = P{3};
          sum_p = 0;
          for i = 1:size(Y,2)
              sum_p = sum_p + l_cross(Y(:,i),P(:,i));
          end
          J = 1/size(X,2)*sum_p + lambda*(sum(sum(W_layers{1}.^2))+sum(sum(W_layers{2}.^2)));
      end


    end
end

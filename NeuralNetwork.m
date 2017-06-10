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
          batch_Size = GDparams{5};
          X_Data = train_data{1};
          Y_Data = train_data{2};
          disp('Starting Mini Batch Gradient Decent');
          fprintf('Epoch = 0');
          if any(strcmpi('BatchNormalize',varargin)) == 1
              set(obj,'BatchNormalize_Var',true);
              X = X_Data;
              for i = 1:(obj.network_size)
                  X = obj.layer_list(i).SetBatchNormalization(X, varargin{:});
              end

          end
          for i = 1:n_Epochs
              train_datasets(obj, X_Data, Y_Data, i, GDparams);
          end
          fprintf('\n');
      end
      function Loss = trainWithLoss(obj, train_data, test_data, GDparams,varargin)
          n_Epochs = GDparams{4};
          batch_Size = GDparams{5};
          Loss = zeros(1,n_Epochs);
          X_Data = train_data{1};
          Y_Data = train_data{2};
          Test_X = test_data{1};
          Test_y = test_data{2};
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
              train_datasets(obj, X_Data, Y_Data, i, GDparams);
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
          % [S,P] = EvaluateClassifier(X_Data, obj.layer_list, obj.network_size, varargin{:});
          G = P{1,obj.network_size} - Y_Data;

          if obj.BatchNormalize_Var
            for i = obj.network_size:-1:2
%                G = obj.layer_list(i).BatchNormBackPass(Y_Data,G, P{i-1}, S{i-1}, GDparams);
               G = obj.layer_list(i).BatchNormBackPass(Y_Data,G, P{i-1}, S{i-1}, GDparams);
            end
            obj.layer_list(1).BatchNormBackPass(Y_Data,G, double(X_Data), X_Data, GDparams);
          else
              for i = obj.network_size:-1:2
                 G = obj.layer_list(i).BackPass(Y_Data,G, P{i-1}, S{i-1}, GDparams);
              end
              obj.layer_list(1).BackPass(Y_Data,G, double(X_Data), X_Data, GDparams);
          end
      end
      function train_datasets(obj, X_Data, Y_Data, epoch, GDparams)

        batch_Size = GDparams{5};
        for j = 1:1:(size(X_Data,2)/batch_Size)
            obj.GradientDecent(X_Data(:,(((j-1)*batch_Size+1):j*batch_Size)), Y_Data(:,(((j-1)*batch_Size+1):j*batch_Size)), GDparams);
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
    end
end

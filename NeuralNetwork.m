classdef NeuralNetwork < matlab.mixin.SetGet

    properties
      layer_list;
      network_size;
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
      function [S,P] = eval(obj, Data)
         [S,P] = obj.EvaluateClassifier(Data, obj.layer_list, obj.network_size);
      end
      function [S,P] = EvaluateClassifier(obj, X)
          layers = obj.layer_list;
          for i = 1:(obj.network_size)
            P{i} = zeros(Layers(i).node_number,size(X,2));
            S{i} = zeros(Layers(i).node_number,size(X,2));
          end

          for i = 1:size(X,2)
             s = Layers(1).eval(X(:,i));
             S{1}(:,i) = s;
             h = Layers(1).activation(s);
             P{1}(:,i) = h;

          end
          if obj.network_size > 1
            for j = 2:2:obj.network_size
              for i = 1:size(X,2)
                 s = Layers(j).eval(P{j-1}(:,i));
                 S{j}(:,i) = s;
                 h = Layers(1).activation(s);
                 P{j}(:,i) = h;
               end
             end
          end
      end
      function J = ComputeCost(obj, X, Y, Layers, lambda)
          [~,P] = EvaluateClassifier(X,Layers);
          P = P{obj.network_size};
          sum_p = 0;
          for i = 1:size(Y,2)
              sum_p = sum_p - log(Y(:,i)'*P(:,i));
          end
          J = 1;%1/size(X,2)*sum_p + lambda*(sum(sum(W_layers{1}.^2))+sum(sum(W_layers{2}.^2)));
      end
      function GradientDecent(obj, X_Data, Y_Data, GDparams)
          [S,P] = obj.eval(X_Data);
          G = P{obj.network_size} - Y_Data;

          for i = obj.network_size:-1:2
             G = obj.layer_list(i).backprop(X_Data, Y_Data,G, P{i-1}, S{i-1}, GDparams);
          end
          obj.layer_list.backprop(X_Data, Y_Data,G, double(X_Data), X_Data, GDparams);
      end

    end
end

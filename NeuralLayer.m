classdef NeuralLayer < matlab.mixin.SetGet
    % A single layer in a neural network
    properties (GetAccess=private)%(GetAccess=private)
        W   % The weight matrix
        b   % The normalising vector
        Activation_function %funcname
        V_W % momentum vectors
        V_b
        act_functions = {'relu', 'softmax'};
    end
    properties
        node_number;
        input_dimension;
        standard_deviation;
    end
    methods
        function obj = NeuralLayer(node_number_input, input_dimension_input,Activation_function_input , standard_deviation)
            obj.input_dimension = input_dimension_input;
            obj.node_number = node_number_input;
            obj.standard_deviation = standard_deviation;

            obj.W = standard_deviation.*double(randn(node_number_input, input_dimension_input));
            obj.b = standard_deviation.*double(randn(node_number_input,1));
            obj.V_W = double(zeros(size(node_number_input, input_dimension_input)));
            obj.V_b = double(zeros(size(node_number_input,1)));



            if any(strcmpi(Activation_function_input, obj.act_functions))
                obj.Activation_function = Activation_function_input;
            else
                error('Activation function is not available in this version' );
            end
        end
        function answer = eval(obj, Data)
          answer = obj.W * double(Data) + obj.b;
%           answer = activation(obj, s);
        end
        function answer = activation(obj, Data)
          if strcmp(obj.Activation_function, 'relu')
            answer = obj.max(0,Data);
          elseif strcmp(obj.Activation_function, 'softmax')
            answer = obj.SOFTMAX(Data);
          else
            % Because of the check when setting activation methods this will
            % never be returned
            answer = Data;
          end
        end
        function ret = SOFTMAX(P)

            e = exp(P);
            one = ones(size(P,1),1);
            split = one'*e;
            ret = e/split(1,1);
        end
        
        function g = backprop(obj, X_Data, Y_Data,G, P, S, GDparams)
          rho = GDparams{1};
          eta = GDparams{2};
          W = obj.W;
          b = obj.b;

          [grad_W, grad_b, g] = obj.ComputeGradients(W, Y_Data,G, P, S, GDparams);
          V_W = obj.V_W;
          V_b = obj.V_b;
          V_W = (rho.*V_W) + eta*grad_W;
          V_b = (rho.*V_b) + eta*grad_b;

          set(obj, 'V_W', V_W);
          set(obj, 'V_b', V_b);
          set(obj, 'W', W - V_W);
          set(obj, 'b', b - V_b);

        end
        function [grad_W, grad_b, g] = ComputeGradients(obj, Y_Data, G, P, S)
          grad_W = 0;
          grad_b = 0;
          g = zeros(size(G));
          for i = 1:size(Y_Data,2)
            grad_b = grad_b + G(:,i);
            grad_W = grad_W + (G(:,i)*P(:,i)');

            next_layers_G = G(:,i)'*obj.W;
            g(:,i) = (next_layers_G*diag((S(:,i)>0)))';
          end
          grad_b = grad_b./size(Y_Data,2);
          grad_W = grad_W./size(Y_Data,2) + 2*lambda*obj.W;
        end
    end

end

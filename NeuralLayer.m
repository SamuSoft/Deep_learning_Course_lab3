classdef NeuralLayer < matlab.mixin.SetGet
    % A single layer in a neural network
    properties (GetAccess=private)
        W   % The weight matrix
        b   % The normalising vector
        Activation_function %funcname
        V_W % momentum vectors
        V_b
        act_functions = {'relu', 'softmax'};

        % BatchNormalization variables
        % ----------------------------
        BatchNormalize_Var = false;
        Train_Data_My;
        Train_Data_v;
        % ----------------------------


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
                obj.Activation_function = lower(Activation_function_input);
            else
                error('Activation function is not available in this version' );
            end
        end
        function s = eval(obj, Data)
            W = obj.W;
            b = obj.b;
            s = W * double(Data) + repmat(b,1,size(Data,2));
        end
        function ret = activation(obj, Data)
          if strcmp(obj.Activation_function, 'relu')
            ret = max(0,Data);
          elseif strcmp(obj.Activation_function, 'softmax')
            %SOFTMAX FUNCTION

            e = exp(Data);
            % one = ones(size(Data,1),1);
            ret = bsxfun(@rdivide,e,sum(e));
            % split = one'*e;
            % ret = e/split(1,1);
          else
            error('Activation function is not available in this version' );
          end
        end
        function sumValue = WeightSum(obj)
            sumValue = sum(sum(obj.W));
        end
        function Value = BatchNormalize(obj, s)
            s = double(s);
            my = obj.Train_Data_My;
            v = obj.Train_Data_v;

            V = (sqrt(diag(v + 0.0000001)))^(-1);
            var = (s - repmat(my,1,size(s,2)));
            Value = V * var;
        end
        function g = BackPass(obj, Y_Data,G, P, S, GDparams)
          rho = GDparams{1};
          eta = GDparams{2};
          W = obj.W;
          b = obj.b;

          [grad_W, grad_b, g] = obj.ComputeGradients(Y_Data,G, P, S, GDparams{3});
          V_W = obj.V_W;
          V_b = obj.V_b;
          V_W = (rho.*V_W) + eta*grad_W;
          V_b = (rho.*V_b) + eta*grad_b;

          set(obj, 'V_W', V_W);
          set(obj, 'V_b', V_b);
          set(obj, 'W', W - V_W);
          set(obj, 'b', b - V_b);
        end
        function [grad_W, grad_b, g] = ComputeGradients(obj, Y_Data, G, P, S, lambda)

            grad_W = 0;
            grad_b = 0;
            W = obj.W;
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
        function Data = SetBatchNormalization(obj, X_Data, varargin)
            Data = obj.activation(obj.eval(X_Data));
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
                s = Data;
                my = (1/size(s,2)).*sum(Data,2);
                set(obj, 'Train_Data_My', my);
                % v = zeros(size(s,1),1);
                % for j = 1:size(s,1)
                %     sum_s = 0;
                %     for i = 1:size(s,2)
                %         sum_s = sum_s + ((s(j,i) - my(j))^2);
                %     end
                %     v(j) = (1/size(s,2))*sum_s;
                % end
                v = var(Data,0,2)*(size(Data,2)-1)/(size(Data,2));
                set(obj, 'Train_Data_v', v);
            end
            % x = obj.activation(obj.eval(Data));
        end
        function ExpMoveBatch(obj, a)
          my = obj.Train_Data_My;
        end
        function W = getW(obj)
            W = obj.W;
        end
        function b = getb(obj)
            b = obj.b;
        end
        function setW(obj,W)
            set(obj, 'W', W);
        end
        function setb(obj,b)
            set(obj, 'b', b);
        end

        function layer = copyLayer(obj)
          layer = NeuralLayer(obj.node_number,obj.input_dimension,obj.Activation_function, obj.standard_deviation);
          layer.setW(obj.W);
          layer.setb(obj.b);
        end
    end
end

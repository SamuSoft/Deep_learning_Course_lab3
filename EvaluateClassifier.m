

function P = EvaluateClassifier(X, W, b)
    P{1} = zeros(50,size(X,2));
    P{2} = zeros(50,size(X,2));
    P{3} = zeros(10,size(X,2));
    for i = 1:size(X,2)
       s1 = W{1}*double(X(:,i)) +b{1};
       P{1}(:,i) = s1;
       h = max(0,s1);
       P{2}(:,i) = h;
       s = W{2}*h + b{2};
       P{3}(:,i) = SOFTMAX(s);
    end

end

function ret = SOFTMAX(P)

    e = exp(P);
    one = ones(size(P,1),1);
    split = one'*e;
    ret = e/split(1,1);
end

%
% function [S,P] = EvaluateClassifier(X, Layers, number_of_layers)
%     for i = 1:(number_of_layers)
%       P{i} = zeros(Layers(i).node_number,size(X,2));
%       S{i} = zeros(Layers(i).node_number,size(X,2));
%     end
%
%     for i = 1:size(X,2)
%        s = Layers(1).eval(X(:,i));
%        S{1}(:,i) = s;
%        h = Layers(1).activation(s);
%        P{1}(:,i) = h;
%
%     end
%     if number_of_layers > 1
%       for j = 2:2:number_of_layers
%         for i = 1:size(X,2)
%            s = Layers(j).eval(P{j-1}(:,i));
%            S{j}(:,i) = s;
%            h = Layers(1).activation(s);
%            P{j}(:,i) = h;
%          end
%        end
%     end
% end
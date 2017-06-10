
function [S,P] = EvaluateClassifier(X, Layers, number_of_layers)
    for i = 1:(number_of_layers)
      P{i} = zeros(Layers(i).node_number,size(X,2));
      S{i} = zeros(Layers(i).node_number,size(X,2));
    end

    for i = 1:size(X,2)
       s = Layers(1).eval(X(:,i));
       S{1}(:,i) = s;
       h = Layers(1).activation(s);
       P{1}(:,i) = h;
       
    end
    if number_of_layers > 1
      for j = 2:2:number_of_layers
        for i = 1:size(X,2)
           s = Layers(j).eval(P{j-1}(:,i));
           S{j}(:,i) = s;
           h = Layers(1).activation(s);
           P{j}(:,i) = h;
         end
       end
    end
end

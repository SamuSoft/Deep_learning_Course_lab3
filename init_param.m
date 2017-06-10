function [args] = init_param(standard_deviation, hidden_layer_nodes_1, d, K)

  % Layer 1
  W_1 = standard_deviation.*randn(hidden_layer_nodes_1, d);
  b_1 = standard_deviation.*randn(hidden_layer_nodes_1,1);

  %  Layer 2
  W_2 = standard_deviation.*randn(K,hidden_layer_nodes_1);
  b_2 = standard_deviation.*randn(K,1);

  W_layers = {W_1,W_2};
  b_layers = {b_1,b_2};

% Initializing momentum vectors
  V_list{1}{1} = zeros(size(W_layers{1}));
  V_list{1}{2} = zeros(size(W_layers{2}));
  V_list{2}{1} = zeros(size(b_layers{1}));
  V_list{2}{2} = zeros(size(b_layers{2}));

  args = {W_layers, b_layers, V_list};


end

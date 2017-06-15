
% [train_X,~,~]                   = LoadBatch('data_batch_1.mat');
[train_X,train_Y,train_y]                   = LoadBatch('data_batch_1.mat');
%     [validation_X,validation_Y,validation_y]    = LoadBatch('data_batch_2.mat');
[test_X,test_Y,test_y]                      = LoadBatch('test_batch.mat');

MyNetwork = NeuralNetwork();



layer1 = NeuralLayer(50,3072,'RELU', .001);

% layer2 = NeuralLayer(20,50,'RELU', .001);

layer3 = NeuralLayer(10,50,'SOFTMAX',.001);

testLayer = NeuralLayer(10,3072,'RELU', .001);

MyNetwork.add(layer1);
% MyNetwork.add(layer2);
MyNetwork.add(layer3);
% MyNetwork.add(testLayer);

% testx = train_X(:,1:100);
% testy = train_Y(:,1:100);

%  Preprocessing data
train_X = double(train_X);
  % ------------------------------------------------
  mean_X = mean(train_X,2);
  train_X = double(train_X) - repmat(mean_X, 1, size(train_X, 2));
  Xtest = double(test_X) - repmat(mean_X, [1, size(test_X, 2)]);
  % ------------------------------------------------

train_size = 100;

train_data = {train_X(:,1:train_size),train_Y(:,1:train_size)};
% train_data = {train_X,train_Y};
test_data = {test_X, test_Y};

rho = .9;
lambda = 9.91095530410570e-07;
eta= 0.0274140702681276;
n_epochs = 20;
batch_size = 100;
% P = MyNetwork.EvaluateClassifier(testx);
GDparams = {rho, eta, lambda, 1, 100};
% GDparams = {.9,.01,.000001,40,100};
% GDparams = {.9,4.1980e-04,3.2764e-06,40,100};
GDparams = {0, eta, lambda, 1, 100};
s = 'BatchNormalize';
rho = 0.9;
batch_size = 20;
net2 = MyNetwork.copyNetwork();
loss1 = MyNetwork.trainWithLoss(train_data, test_data, GDparams,s);
loss2 = net2.trainNum(train_data, test_data, GDparams,s);
% MyNetwork.trainWithLoss(train_data, test_data, GDparams,s);

[W_num, b_num]=net2.getWb();
[W_an, b_an] = MyNetwork.getWb();

for i = 1:2
    errW(i) = ErrDiff(W_num{i}, W_an{i});
    errb(i) = ErrDiff(b_num{i}, b_an{i});
end
% e_min = -10;
% e_max = log(2);
% l_min = -10;
% l_max = log(6);
% for i = 1:20
%
%       MyNetwork = NeuralNetwork();
%       layer1 = NeuralLayer(50,3072,'RELU', .001);
%       layer2 = NeuralLayer(20,50,'RELU', .001);
%       layer3 = NeuralLayer(10,20,'SOFTMAX',.001);
%       MyNetwork.add(layer1);
%       MyNetwork.add(layer2);
%       MyNetwork.add(layer3);
%       % random eta
%       e = e_min + (e_max-e_min)*rand();
%       eta = 10^e;
%       l = l_min + (l_max-l_min)*rand();
%       lambda = 10^l;
% %       lambda = 9.91095530410570e-07;
%
% %        eta= 0.0274140702681276;
%
% %       eta = .01;
% %       lambda = .000001;
% %       rho = .9;
%       n_epochs = 5;
% %       hidden_layer_nodes_1 = 50;
%       GDparams = {rho, eta, lambda, n_epochs, batch_size};
% %       GDparams = {n_batch, eta, n_epochs, rho, standard_deviation, lambda, hidden_layer_nodes_1};
%       Loss = MyNetwork.trainWithLoss(train_data, test_data, GDparams,s);
%       loss_list_test{i} = Loss;
%       lambda_list(i) = lambda;
%       eta_list(i) = eta;
%       disp('Loop: ');
%       disp(i);
%   end

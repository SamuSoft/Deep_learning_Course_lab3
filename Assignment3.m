
% [train_X,~,~]                   = LoadBatch('data_batch_1.mat');
[train_X,train_Y,train_y]                   = LoadBatch('data_batch_1.mat');
%     [validation_X,validation_Y,validation_y]    = LoadBatch('data_batch_2.mat');
[test_X,test_Y,test_y]                      = LoadBatch('test_batch.mat');

MyNetwork = NeuralNetwork();



layer1 = NeuralLayer(50,3072,'RELU', .001);

layer2 = NeuralLayer(20,50,'RELU', .001);

layer3 = NeuralLayer(10,20,'SOFTMAX',.001);

testLayer = NeuralLayer(10,3072,'RELU', .001);

MyNetwork.add(layer1);
MyNetwork.add(layer2);
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

train_data = {train_X(:,1:100),train_Y(:,1:100)};

test_data = {test_X, test_Y};

% layer1.eval(testx);

% P = MyNetwork.EvaluateClassifier(testx);
% GDparams = {rho, eta, lambda, n_epochs, batch_size}
GDparams = {.9,.01,.000001,40,100};

% MyNetwork.ComputeCost(test_X,test_Y, GDparams{3})
% MyNetwork.train(testx, testy, GDparams);
% MyNetwork.ComputeCost(test_X,test_Y, GDparams{3})
s = 'BatchNormalize';

Loss = MyNetwork.trainWithLoss(train_data, test_data, GDparams,s);

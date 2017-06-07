
% [train_X,~,~]                   = LoadBatch('data_batch_1.mat');
[train_X,train_Y,train_y]                   = LoadBatch('data_batch_1.mat');
%     [validation_X,validation_Y,validation_y]    = LoadBatch('data_batch_2.mat');
% [test_X,test_Y,test_y]                      = LoadBatch('test_batch.mat');

MyNetwork = NeuralNetwork();

layer1 = NeuralLayer(50,3072,'RELU', .01);

layer2 = NeuralLayer(10,50,'SOFTMAX', .01);

MyNetwork.add(layer1);
MyNetwork.add(layer2);

testx = train_X(:,1:100);
testy = train_Y(:,1:100);

P = MyNetwork.eval(test);
GDparams = {1,1,1};
MyNetwork.GradientDecent(testx, testy, GDparams)
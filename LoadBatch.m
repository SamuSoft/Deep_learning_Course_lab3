%  X contains the image pixel data, has size d?N, is of type double or 
%  single and has entries between 0 and 1. N is the number of images (10000) 
%  and d the dimensionality of each image (3072=32?32?3).
%  
%  Y is K?N (K= # of labels = 10) and contains the one-hot representation of 
%  the label for each image.
%  
%  y is a vector of length N containing the label for each image. A note of 
%  caution. CIFAR-10 encodes the labels as integers between 0-9 but Matlab 
%  indexes matrices and vectors starting at 1. Therefore it may be easier to 
%  encode the labels between 1-10.
function [X, Y, y] = LoadBatch(filename)
    addpath '~/Documents/Courses/Deep Learning/Assignment/1/Datasets/cifar-10-batches-mat';
    A = load(filename);
    X = A.data';
    X = X./255;
    % Change unique values to one-hot later
    Y = zeros(size(unique(A.labels),1),size(A.labels,1));
    y = A.labels + 1.;
    
    for i = 1:size(A.labels,1)
        Y(y(i),i) = 1;
    end
end
    
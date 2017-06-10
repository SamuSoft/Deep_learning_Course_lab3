function ret = SOFTMAX(P)

    e = exp(P);
    one = ones(size(P,1),1);
    split = one'*e;
    ret = e/split(1,1);
end

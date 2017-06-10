function diff = ErrDiff(X,Y)
  diff = sum(abs(max(X - Y)))/max(.001,sum(abs(max(X))) +sum(abs(max(Y))));
end

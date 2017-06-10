% % b = {ones(size(a{1})),ones(size(a{1})),ones(size(a{1}))};
% zero = zeros(1,10);
% 
% for j = 1:700
%     if sum(isnan(a{1}{1,j})) > 0
%         b{1}{j} = a{1}{1,j};
%         b{2}{j} = a{2}{1,j};
%         b{3}{j} = a{3}{1,j};
%     else
%         b{1}{j} = zero;
%         b{2}{j} = zero;
%         b{3}{j} = zero;
%         
%     end
% end
% scatter3(b{1},b{2},b{3});
% xlabel('loss');
% ylabel('lambda');
% zlabel('eta');
hold on
plot(1:40, real(Loss))
xlabel('Epochs')
ylabel('Loss')
title('2-layers, 50 hidden nodes')
hold off
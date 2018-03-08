clear ; close all; clc

x = [50, 60, 70, 80, 90];
y = [25.8, 34.9, 48.2, 66.8, 81.1];

plot(x, y, 'x');
hold on

% y = a * exp(b*x)
% log(y) = b*x + log(a)

p = polyfit(x, log(y), 1);
b = p(1);
a = exp(p(2));

X = [0:1:100];
Y = a .* exp(b.*X);

plot(X,Y);

% (b)
yb = a * exp(b * 0);
fprintf('(b) 1900年估计人口为%f (实际人口为13607272)\n', yb)

% (c)
yc = exp(p(1)) * 100 - 100;
fprintf('(c) 年增长率为%.2f%% \n', yc)

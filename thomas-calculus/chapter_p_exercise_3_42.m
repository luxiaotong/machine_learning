clear ; close all; clc

x = [4, 11, 21, 36, 46, 51, 60, 70, 80];
y = [5.2, 6.0, 6.9, 9.6, 11.4, 12.7, 16.0, 18.3, 20.6];

plot(x, y, 'x');

hold on

% y = a * exp(b*x)
% log(y) = b*x + log(a)
% log(y) = p(1)x + log(exp(p(2)))

p = polyfit(x, log(y), 1);
b = p(1);
a = exp(p(2));

X = [0:1:100];
Y = a .* exp(b.*X);

plot(X, Y);
hold on


% log(y) = p(1)x + p(2)

X = [0:1:100];
Y2 = exp(p(1) .* X .+ p(2));
plot(X, Y2);

% 所以Y和Y2是等价的


% (b)
yb = exp(p(1) * 90 + p(2));
fprintf('(b) 1990年估计南非人口为%f \n', yb)

% (c)
yc = exp(p(1)) * 100 - 100;
fprintf('(c) 年增长率为%.2f%% \n', yc)

clear ; close all; clc

% x = -2*pi:0.1:2*pi;
% y = (4.*x) ./ (x.^2 +1);
% 
% plot(x, y);

hold on;

x = -2*pi:0.1:2*pi;
y = 2*sin(2*atan(x));

plot(x, y);

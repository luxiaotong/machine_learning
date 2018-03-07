clear ; close all; clc

x = [0,3,6,7,9,13,17,19,23,27,30];
y = [6,8,10,13,15,20,22,25,29,32,33];
p = polyfit(x, y, 1);

plot(x, y, 'x');
hold on


X = [0:1:60];
Y = p(1) .* X .+ p(2);

plot(X, Y);

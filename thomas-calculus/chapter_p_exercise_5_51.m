clear ; close all; clc

x = 1:1:6;
y = [3.42, 0.73, 0.12, 2.16, 4.97, 5.97];

plot(x, y, 'rx');
hold on;

X = 1:0.1:6;
a = 3;
b = pi/3;
c = -3*pi/2;
d = 3;
Y = a * sin(b * X + c) + d;

plot(X, Y);

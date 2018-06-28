clear ; close all; clc

x = 1:1:12;
y = [34, 30, 39, 44, 58, 67, 78, 80, 72, 63, 51, 40];

plot(x, y, 'rx');
hold on;

X = x;

a = 25;
b = pi/6;
h = 5;
k = 55;
Y = a * sin(b * (X - h)) + k;

plot(X, Y);

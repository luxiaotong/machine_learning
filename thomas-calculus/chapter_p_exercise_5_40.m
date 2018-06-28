clear ; close all; clc

x = [-3/2*pi:0.1:3/2*pi];
plot(x, cos(x));

hold on;

x1 = -pi/3+0.01:0.01:-pi/2-0.01;
x2 = -pi/2+0.01:0.01:pi/2-0.01;
x3 = pi/2+0.01:0.01:(3*pi/2)-0.01;
plot(x1, sec(x1), x2, sec(x2), x3, sec(x3));

grid on;

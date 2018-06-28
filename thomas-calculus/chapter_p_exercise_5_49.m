clear ; close all; clc

data = importdata('./chapter_p_exercise_5_49.txt',',');
x = data(:, 1);
y = data(:, 2);

% plot(x, y, 'x');
% hold on
% 
% 
% func_sin = @(a,t) a(1)*sin(2*pi*t/a(2) + a(3)) + a(4);
% A= lsqcurvefit( func_sin, [5000 0 25000 0], x, y);
% 
% plot(x, func_sin(A, x));

yu = max(y);
yl = min(y);
yr = (yu-yl);                               % Range of ‘y’
yz = y-yu+(yr/2);
zx = x(yz .* circshift(yz,[0 1]) <= 0);     % Find zero-crossings
per = 2*mean(diff(zx));                     % Estimate period
ym = mean(y);                               % Estimate offset
fit = @(b,x)  b(1).*(sin(2*pi*x./b(2) + 2*pi/b(3))) + b(4);    % Function to fit
fcn = @(b) sum((fit(b,x) - y).^2);                              % Least-Squares cost function
s = fminsearch(fcn, [yr;  per;  -1;  ym])                       % Minimise Least-Squares
xp = linspace(min(x),max(x));
figure(1)
plot(x,y,'b',  xp,fit(s,xp), 'r')
grid

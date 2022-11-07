close all; 
clc;
clear all;

X = [1 1; -1 1; 1 -1; -1 -1;  2 2; -2 2; 2 -2; -2 -2;  ];
X_F = map_feature(X);

figure(1);
grid on;
xlabel('X_1');
ylabel('X_2');
axis([-5 5 -5 5]);
hold on;
plot(X(1:4,1),X(1:4,2), 'bo');
plot(X(5:8,1),X(5:8,2), 'ro');
hold off;

figure(2);
grid on;
xlabel('X_1');
ylabel('X_2');
axis([-15 0 -15 0]);
hold on;
plot(X_F(1:4,1),X_F(1:4,2), 'bo');
plot(X_F(5:8,1),X_F(5:8,2), 'ro');
hold off;




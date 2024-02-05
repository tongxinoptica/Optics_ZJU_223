close all
clc

w = 2;%Gauss光斑半径
m1 = 4;
m2 = 4;
n1 = 2;
n2 = 4;   %the phase factor

pix_x=1920;
% pix_y=1080;
pix_y=1200;
dx=8*10^(-6);
dy=dx;
x=(-pix_x*dx/2:dx:(pix_x/2-1)*dx);
y=(-pix_y*dy/2:dy:(pix_y/2-1)*dy);

[xx, yy]=meshgrid(x,y);

% E = zeros(pix_x, pix_y);
x_y = exp(-(xx.^2+yy.^2)/w^2);
xy_vector = xx + 1i*yy;
phi = angle_1(xy_vector);
Psi = 2*pi*(mod(floor(m1*phi/2/pi- 1),2).*(rem(m1*phi,2*pi)/2/pi).^n1+mod(floor(m2*phi/2/pi),2).*(rem(m2*phi,2*pi)/2/pi).^n2);
E = x_y.*exp(1i*Psi);

Phase = angle_1(E);

figure;
% surf(xx,yy,Phase);
% shading interp   
% colormap gray
% view(90,90);
% axis off

imshow(Phase,[],'border','tight')
colormap gray

%Phase=mat2gray(Phase);

function phi = angle_1(xy_vector)
phi = mod(angle(xy_vector),2*pi);
end
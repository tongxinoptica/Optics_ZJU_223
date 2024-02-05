clc;
tic
A0=1;
w=0.5;
m1=4;
m2=4;
n1=2;
n2=4;
n3=1;
t=2;
z=1.2e3;
lamda=632e-6;k=2*pi/lamda;
N=1000;
K1=6;
ax=2*K1/N;%
bx=-2*K1/N-K1;
viewport=1.3;
indexrange=(-floor(viewport/ax):floor(viewport/ax))+floor(N/2);
x=linspace(-K1,K1,N+1);
[X,Y]=meshgrid(x);
Lside=ax*N;
kx=(-N/2:N/2)/Lside;
[kx1,ky1]=meshgrid(kx);%2*pi*(mod(floor(1/3/pi*(m1*phi+4*pi)),2).*mod(floor(1/3/pi*(m1*phi+4*pi)-1/3),2).*(rem(m1*phi,2*pi)/2/pi).^n1+mod(floor(1/3/pi*(m1*phi+2*pi)),2).*mod(floor(1/3/pi*(m1*phi+2*pi)-1/3),2).*(rem(m1*phi,2*pi)/2/pi).^n2+mod(floor(1/3/pi*(m1*phi+0*pi)),2).*mod(floor(1/3/pi*(m1*phi+0*pi)-1/3),2).*(rem(m1*phi,2*pi)/2/pi).^n3)

xy=exp(-(X.^2+Y.^2)/w^2);%高斯项(sin(rem(m1*phi,2*pi)/2-pi/2)+1)*m2*pi
xy_vector = X + 1i*Y;%复数1*pi*(sin(rem(4*phi,4*pi)/4+pi/2))
phi=angleNormalized(xy_vector);
r=(X.^2+Y.^2).^0.5+eps;
    Psi_1 =2*pi*(mod(floor(m1*phi/2/pi-1),2).*(rem(m1*phi,2*pi)/2/pi).^n1+mod(floor(m2*phi/2/pi),2).*(rem(m2*phi,2*pi)/2/pi).^n2);%计算每一个复数对应的相位角2*pi*(rem(l*phi,2*pi)./(2*pi)).^n 
E1=A0*xy.*exp(1i*Psi_1);%写出电场的表达式2*pi*(mod(floor(m1*phi/2/pi-1),2).*(rem(m1*phi,2*pi)/2/pi).^n1+mod(floor(m2*phi/2/pi),2).*(rem(m2*phi,2*pi)/2/pi).^n2)
I1=abs(E1)^2;%光强
Phase1=angleNormalized(E1);%相位角（其实这个不就是PSI1吗）
E2=fftshift(fft2(E1));%第一次傅里叶变换
kxky=exp(1i*(kx1.^2+ky1.^2)*z/(2*k));

E3=E2.*kxky;
E4=ifft2(fftshift(E3)).*exp(1i*k*z);%傅里叶逆变换
[Fx, Fy]=gradient(E4);
[fx, fy]=gradient(conj(E4));
px=E4.*fx-conj(E4).*Fx;
py=E4.*fy-conj(E4).*Fy;
j=1i*(X.*py-Y.*px);
I2=E4.*conj(E4);%光强E4.*conj(E4)
J2=(j.*conj(j)).^0.5;
S1=sum(I2(:));
S2=sum(j(:));
S3=S2/S1;
J2=J2./max(max(J2));
I2Prime=I2(indexrange,indexrange);
J2Prime=J2(indexrange,indexrange);
Phase4=angleNormalized(E4);%计算相位角
Phase4Prime=Phase4(indexrange,indexrange);
Phase1Prime=Phase1(indexrange,indexrange);
% figure;%画图
% subplot(2,2,1);
x=X(indexrange,indexrange);
y=Y(indexrange,indexrange);
pcolor(x,y,I2Prime);
shading interp;
colormap jet;
title("OAM, z="+z/1e3+"m");
% subplot(2,2,2);
pcolor(x,y,I2Prime);
shading interp;
colormap jet;
title("Intensity, z="+z/1e3+"m");

function phi=angleNormalized(xy_vector)
         phi= mod(angle(xy_vector),2*pi);
end

clc
clear
image = im2double(imread('D:\HR_data\HR_Binary\10.jpg'));
image = double(imresize(image,[2000,2000]));
image = image/max(max(image)).^0.5; 
imagesc(image)

%%
N = 2000;
lambda = 532e-6;               %波长1064nm
x = linspace(-3,3,N); 
y = linspace(-3,3,N);
[X,Y] = meshgrid(x,y);
[theta,r] = cart2pol(X,Y);
w = 5;                          %高斯光束束腰宽度
k = 2*pi/lambda;                %波数
k_r = 10;                       %径向波矢 - 常量
k_z = sqrt(k^2-k_r^2);          %轴向波矢
z = 100;
for n = 1 : 1                    %贝塞尔函数阶数n = 0,1,2,3等等
    E = image.*besselj(n,k_r*r).*exp(-r.^2/w^2).*exp(1i*n*theta);
    I = E.*conj(E);
    I = I/max(max(I));          %归一化
    figure;
    imagesc(x,y,I)
    set(gca,'fontname','times new roman');
    title('贝塞尔-高斯光束');
    xlabel('x/mm','fontname','times new roman');
    ylabel('y/mm','fontname','times new roman');
    
    dx = x(2) - x(1);
    df = 1/(N*dx);
    fX = (-N/2:N/2-1) * df;
    fY = (-N/2:N/2-1) * df;
    [Fx, Fy] = meshgrid(fX, fY);
    H = exp(-1i*pi*lambda*z*(Fx.^2 + Fy.^2)); % 传播函数
    BG_F = fftshift(fft2(ifftshift(E)));
    BG_propagated = fftshift(ifft2(ifftshift(BG_F .* H)));
    figure;
    imagesc(x, y, abs(BG_propagated).^2);
    xlabel('x (m)');
    ylabel('y (m)');
    title('一阶贝塞尔-高斯光束在自由空间中传播后的强度分布');


    
%     for a=1:N
%         for b=1:N
%             E2(a,b) = -1i/lambda/z*exp(1i*k*z)*sum(sum(E.*exp(1i*k/2/z.*((x(a)-X).^2+(y(b)-Y).^2))));
%         end
%     end
%     I2 = E2.*conj(E2);      I2 = I2/max(max(I2));
%     figure(2);
%     imagesc(x,y,I2)
%     set(gca,'fontname','times new roman','fontsize',16);
%     title([num2str(n),'阶贝塞尔-高斯光束自由传输后光强分布'],'fontname','华文中宋','fontsize',16);
%     xlabel('x/mm','fontname','times new roman','fontsize',16);
%     ylabel('y/mm','fontname','times new roman','fontsize',16);
end


function [noise_img, output_img, ft_pad_img, fil_img] = taskA(filePathString)
    
    % task a
    % Read the input image taskA.png and convert it to a grayscale image (double values between 0.0 and 1.0)
    I = double(imread(filePathString)) / 255;
    gray_img = rgb2gray(I);
    
    % task b
    M = 0;
    V = 0.01;
    noise_img = imnoise(gray_img, 'gaussian', M, V);
    whos noise_img;
    
    % task c
    
    % gaussian filter
    sigma = 1.0; %standard deviation
    
    sz = 5; %window size
    ind = -floor(sz/2) : floor(sz/2); 
    [x, y] = meshgrid(ind, ind);
    
    M = size(x,1) - 1;
    N = size(y,1) - 1;
    
    Exp_comp = -(x.^2+y.^2)/(2*sigma*sigma);
    Kernel= exp(Exp_comp)/(2*pi*sigma*sigma);
    Kernel = Kernel / sum(Kernel(:))
    
    %Initialize
    output_img=zeros(size(noise_img));
    
    %Pad the vector with zeros
    padded_noise_img = padarray(noise_img,[sz sz]);

    %Convolution
    for i = 1:size(noise_img,1)-M
        for j =1:size(noise_img,2)-N
            temp = noise_img(i:i+M,j:j+M).*Kernel;
            output_img(i,j)=sum(temp(:));
        end
    end    
    
    g = fspecial('gaussian', size(padded_noise_img), 10);
      
    ft_img = fft2(noise_img);
    size(ft_img)
    ft_pad_img = fft2(padded_noise_img);
    size(ft_pad_img)
%    fil_img = ft_img .* ft_pad_img;
    
    figure('name', 'ft_img'), imshow(ft_img);
    figure('name', 'ft_pad_img'), imshow(ft_pad_img);
    
%    fil_img = ifft2(fil_ft);
    
    % ft_Centred = fftshift(ft_img); 
        
    %FFT:
    % F(u,v) = 1/M*N SUM(0,M-1)SUM(0,N-1) f(x,y)*e^(j*2*pi*(u*x/M + l*y/N))
    % for every u, v
    
    % scalar = magnitude -> abs(comlex_value)
    
    %centering
    
    %log
    
    %apply Gauss2D Filter
   

end

function gauss2dFilter(noise_img, sigma)
    %term_1 = (1 / (2*pi*sigma^2));
    %term_2 = (x^2 + y.^2) / (2*sigma^2);
    %gauss2D = term_1.*exp( -term_2 );
    
    % add padding
end

function ifft2(noise_img)

end


function [H, index_theta, index_rho] =  houghLineDet(image, im_x, im_y)
   
    % output H = voting array, index_array the ranges of theta and rho(p) 
    %task c
    %apply threshold to image
    BW = im2bw(image, graythresh(image));
    figure('name', 'thresholded'), imshow(BW, []);
    
    % initialize index vectors
    rho_max = round(sqrt(size(BW,1)^2+size(BW,2)^2));
    index_theta = [-90: 1: 89]; %define theta range
    index_rho = [-rho_max: 1: rho_max]; %define rho range
    
    % initialize voting array
    num_cols = 181;
    num_rows = 2*rho_max+1;
    H = zeros(num_rows, num_cols);
    for x = 1:size(BW,1)
        for y = 1:size(BW,2)
            % if pixel is white -> edge detected
            if BW(x,y) == 1 
                % gradient detection
                theta = round(atand(im_y(x,y)/im_x(x,y)));
                rho = round(x*cosd(theta)+y*sind(theta));
                % cast to int
                h_theta = round(theta) + 91;
                h_rho = round(rho) + rho_max;
                
                H(h_rho,h_theta) = H(h_rho, h_theta) + 1;
            end
        end
    end
    
    % Task e to g
    % find local maxima with MATLAB function houghpeaks

    figure, imshow (H, [],"XData",index_theta,"YData",index_rho);
    title ("Hough transform of edge image \n 20 peaks marked");
    axis on; xlabel("theta [degrees]"); ylabel("rho [pixels]");
    peaks = houghpeaks (H, 50);
    peaks_rho = index_rho(peaks(:,1))
    peaks_theta = index_theta(peaks(:,2))
    hold on;
    plot(peaks_theta,peaks_rho,"sr");
    hold off;
    
    % Task h to i
    I = imread('img/input_ex3.jpg');
    lines = houghlines(BW,index_theta,index_rho,peaks,'FillGap',5,'MinLength',7);
    figure, imshow(I), hold on
    max_len = 0;
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

        % Plot beginnings and ends of lines
        plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
        plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

        % Determine the endpoints of the longest line segment
        len = norm(lines(k).point1 - lines(k).point2);
        if ( len > max_len)
            max_len = len;
            xy_long = xy;
        end
    end
    

end
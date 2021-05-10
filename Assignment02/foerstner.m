function foerstner(originImage, I_x, I_y)
% input is original image and the two gradient images Ix and Iy
   I_x2 = I_x.^2;
   I_y2 = I_y.^2;
   I_xy = I_x.*I_y;
   
% a compute autocorrelation matrix (from 3 to size-3 because borders)

   W = zeros(size(originImage,1), size(originImage,2));
   Q = zeros(size(originImage,1), size(originImage,2));
     
    for x = 3:size(originImage,1)-3
        for y = 3:size(originImage,2)-3
            % autocorrelation matrix
            M = compute_autocorr_matrix(I_x2, I_y2, I_xy, x,y);
            %cornerness
            w = trace(M)/2-sqrt((trace(M)/2)^2-det(M));
            W(x,y) = w;
            %roundness
            q = (4*det(M))/(trace(M)^2);
            Q(x,y) = q;
        end
    end
    
    % b compute cornerness and roundness

     figure('name', 'W'), imshow(W, []),colormap('jet')

     figure('name', 'Q'), imshow(Q, []),colormap('jet')
     


% c derive bianry mask

% d plot
end

function M = compute_autocorr_matrix(I_x2, I_y2, I_xy, x, y)
    % wn  = 1 matrix -> just multiply every value times 1
    M = 0;
    for i = x-2:x+2
           for j = y-2:y+2
                M = M + 1*[I_x2(i,j), I_xy(i,j);
                         I_xy(i,j),I_y2(i,j)];
           end
    end
       
 
           
        
        
        
    
end
function GoG(image)
    % 1. Define standard deviation
    sigma = 0.5;

    % 2. Filter kernel radius
    r = cast(abs(3*sigma),'int32');       % abs  =  absoulte value
    
    % 3. define c_x and c_y arrays with (? * 2 + 1) columns and rows for
    % centered local coordinates:
    
    c_x = zeros(r*2+1,r*2+1);

    for row = 1:r*2+1
        for col = 1:r*2+1
            c_x(row,col) = cast(-(size(c_x,1)/2), 'int32') + col;
        end
    end
    
    c_y = c_x';

    % c_x = [-2,-1,0,1,2;
    %        -2,-1,0,1,2;
    %       -2,-1,0,1,2;
    %       -2,-1,0,1,2;
    %        -2,-1,0,1,2;]
    
    % 4. compute filter using c_x and c_y for x and y
    
    GoG_filter_x = gradient(c_x,c_y, sigma);
    GoG_filter_y = gradient(c_y,c_x, sigma);
    
end

function [grd_x1] = gradient(x, y, sigma)
    %input: x and y, sigma = mask radius
    grd_x = (x/2*pi*sigma^4)
    grd_x1 = grd_x*exp(-(x.^2+y.^2)/2*sigma^2)
    
end
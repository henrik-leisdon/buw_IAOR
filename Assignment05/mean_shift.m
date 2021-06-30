function [image] = mean_shift(I) 
    
    % Task B: The effect of window size
    % The large windoe size is the less number of the clustering would be found. Hence, with larger window size, more details(colours) in the image segmentation can be seen. In addition, to reduce the computing expense we tried to adjust the window size (the larger the window size is the less of computing time)
    
    % Task D: Compare the results produced with the algorithms from k mean and mean shift
    % Unlike the K-Means cluster algorithm, mean-shift does not require specifying the number of clusters in advance. Hence, more details (e.g colours) in the image may be reserved by the window size in the mean shift algorithm, or the pre-define clustering number in the k mean alorithm
    
    % Task E: How to accelerate the operation of mean shift?
    % Although we are not able to generate the desired result, during the implementation, we tried to only use sample of the feature points(e.g start with every 1000th pixel as a centroid, if the window is large enough it will probably cluster every pixel). The mean algorithm’s runtime complexity is O(KN2), , where N is the number of datapoints and K is the number of iterations of Meanshift. Another approach is to limit the number of iterations.
    
    % Note: We are not able to complete the algorithm, however please have a look at our rationale by the comments below for our logical thining process
    
    % build feature space based on R G B
    % R | G | B | X | Y
    
    data_points = I;

    whos data_points
    
    % plot the origin image
    figure;
    subplot(1, 2, 1);
    title("Origin image");
    
    % visualise the origin feature space
    figure;
    subplot(1, 2, 1);
    scatter3(I(:, 1), I(:, 2), I(:, 3), 12, I(:, 1 : 3)); 
    title("Origin feature space");
  
    % Pre-define varaibles
    
    % Window size, which will determine the neighbors within the window / kernel, it need to be adjusted accordingly
    window_size = 0.2; 
    
    % Sampling window number 
    sample_size = 5;
        
    % initialize centroids and its window
    % based on the visualization of detected groups in feature space, either we manually assign local sample data points to save the cost of the operation or randonly assign the data point value
    
    center_list = [0.2 0.2 0.2; 0.4 0.4 0.4; 0.6 0.6 0.6; 0.8 0.8 0.8;  ]; % manually assign
    
%    for i = 1:sample_size % randonly assign
%        
%        max = size(data_points,1);
%        r = randi([1,max],1);
%        center_list(i,1) = data_points(r,1);
%        center_list(i,2) = data_points(r,2);
%        center_list(i,3) = data_points(r,3);
%        
%    end
    
    for i = 1 : size (center_list, 1) % first loop for each data point to find its mode
        
        % have a while loop to determine the convergence meets the condition
        % For each iteration, we find the centroid of all the points in the window, shift the window accordingly, and repeat. However, after a sufficient number of steps, the position of the centroid of all the points, and the current location of the window will coincide 
        while 1
            
            % find all the data points within the window
            weightXYZ = zeros(1, 3); % create 1 by 3 matrix to store the weight of data points within the window
            nm_dps = 0; % number of data points within the window
            
        
            for j = 1 : size (data_points, 1) % the second loop to find all the data points within the window
        
                window_center = center_list(i, :); %window center
                possible_dp = data_points(j, 1:3); %possible data point
        
                distance = calculate_distance(possible_dp, window_center); % debug
        
                if (distance < window_size)
                
                    weightXYZ =  weightXYZ + possible_dp;
                    nm_dps = nm_dps + 1;
                
                end %end if  
         
            end % end second loop
        
            % each iteration in while loop would shift window towards local density maximum for each centroid in the center list 
            % compute the mean
            new_centroid = weightXYZ / nm_dps;
    
            % whether or not to shift the window or convrgence occur
            if isequal(new_centroid, center_list(i, 1:3))
                
                break
                disp('break while loop')
   
            end
        
            % shift the window
            center_list(i, 1:3) = new_centroid;
             
        end % end while
                  
    end % end first loop
    
    % merge centroids: remove duplicate element in the center list
    final_centroid_list = unique( center_list,'rows')
    
    % assign the centroid value to each data points
    for n = 1 : size(data_points, 1)
        
        distance_first = calculate_distance(data_points(n, 1:3), final_centroid_list(1,:));
        distance_second = calculate_distance(data_points(n, 1:3), final_centroid_list(2,:));
        
        if distance_first > distance_second
            
            data_points(n, 1:3) = final_centroid_list(2,:)
            
        else
            data_points(n, 1:3) = final_centroid_list(1,:)
            
        end
        
    end
    
    image = zeros(600,600,3);
   
    
    for i = 1:size(data_points,1)
        
        image(data_points(i,4),data_points(i,5),1) = data_points(i, 1);
        image(data_points(i,4),data_points(i,5),2) = data_points(i, 2);
        image(data_points(i,4),data_points(i,5),3) = data_points(i, 3);
    end

    
    image = data_points;
    
    % visualise the feature space after mean shift 
    subplot(1, 2, 2); 
    scatter3(data_points(:, 1), data_points(:, 2), data_points(:, 3), 12, data_points(:, 1 : 3));
    title("feature space after mean shift");
        
  
end % end mean_shift function

function [distance] = calculate_distance(A,B)
  result = B-A;
  distance = sqrt( result(1)^2 + result(2)^2 + result(3)^2);
end 
 
 
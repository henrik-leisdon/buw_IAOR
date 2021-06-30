function  main()
    
    
    
% main function for assignment 5

    clc all;
    close all;
    clear;
    
    pkg load image;
    pkg load statistics;
    
    [image] = loadAndConvert('img/inputEx5_1.jpg');
    % [image] = loadAndConvert('img/inputEx5_2.jpg');
    
    %figure
    %scatter3(image(:,1),image(:,2), image(:,3),'.','r')
    [out_imgA] = k_means(image, 4);
    saveImage (out_imgA, 'task_A_img_4')

    %Task B

    output = mean_shift(image);
    saveImage (output, 'task_B_input2')

    %mean_shift(image);

end


function [out] = loadAndConvert (fileName)

    %loads the image
    I = im2double(imread(fileName));
    
    %Reshape the image to 3D feature space (L*U*V)??
    F = reshape(I,[],3);
    
    % Creating the a zero matrix for x and y
    coord = zeros(size(I,1)*size(I,2),2);
    
    % Setting the RGB points to the x and y position
    it = 1;
    x = 1;
    for i=1:size(I,1)
        y = 1;
        for j=1:size(I,2)
            
            coord(it,1) = y;
            coord(it,2) = x;
            y = y+1;
            it = it+1;
        end
        x = x+1;
            
    end
    
    %coord
    
    % outputs a matrix of the form:
    % R | G | B | x | y
    out = [F coord];
   	
    
 
end
function saveImage (image, filename)
  
  % saves the image to a jpg. Only the filename is needed.
  file = strcat(filename, '.jpg');
  imwrite(image, file, 'jpg');
 
end

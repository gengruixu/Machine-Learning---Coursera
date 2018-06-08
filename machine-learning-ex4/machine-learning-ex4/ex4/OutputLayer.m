function y_new = OutputLayer(y,num_labels) 
    num_example = size(y,1);
    num_class = num_labels;
    y_new = zeros(num_example, num_class);
    for i=1:num_example
        y_new(i,y(i)) = 1;
    end
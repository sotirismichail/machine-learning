function retval = map_feature(X)

    [n,m] = size(X);
    retval = zeros(n,m);

    for i=1:n
        retval(i,1) = X(i,1) - (norm(X(i,:))^2) -4;
        retval(i,2) = X(i,2) - (norm(X(i,:))^2) -4; 
    end

end

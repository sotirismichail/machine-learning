function v = fisherLinearDiscriminant(X1, X2)

    nfeatures = size(X1, 2);
    m1 = size(X1, 1);
    m2 = size(X2, 1);

    for i = 1:nfeatures
      mu1(i) = sum(X1(:,i))*(1/m1); % mean value of X1
    end
    mu1 = mu1';

    for i = 1:nfeatures
      mu2(i) = sum(X2(:,i))*(1/m2); % mean value of X2
    end
    mu2 = mu2';

    S1 = (1/m1).*(transpose(X1) * X1); % scatter matrix of X1
    S2 = (1/m2).*(transpose(X2) * X2); % scatter matrix of X2

    p1 = m1/(m1+m2);  %A priori probabilities
    p2 = m2/(m1+m2);

    Sw = p1*S1 + p2*S2; % Within class scatter matrix

    v = inv(Sw)*(mu1-mu2); % optimal direction for maximum class separation

    v = v/norm(v); % return a vector of unit norm

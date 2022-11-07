function A = myLDA(Samples, Labels, NewDim)
% Input:
%   Samples: The Data Samples
%   Labels: The labels that correspond to the Samples
%   NewDim: The New Dimension of the Feature Vector after applying LDA

	  [NumSamples NumFeatures] = size(Samples);
    NumLabels = length(Labels);
    if(NumSamples ~= NumLabels) then
        fprintf('\nNumber of Samples are not the same with the Number of Labels.\n\n');
        exit
    end
    Classes = unique(Labels);
    NumClasses = length(Classes);
    totalSamples = size(Samples, 1);

    Sw = zeros(NumFeatures, NumFeatures);
    Sb = zeros(NumFeatures, NumFeatures);
    classSamples = zeros(1, NumClasses);

    % For each class i
	  % Find the necessary statistics
    % Calculate the Class Prior Probability
    for i = 1:NumClasses
      classSamples(i) = sum(Labels == Classes(1));
      P(i) = classSamples(i)/totalSamples;
    end

    % Calculate the Class Mean
    for i = 1:NumClasses
      samplesOfClass = Samples(Labels == Classes(i), :);
      mu(i,:) = mean(samplesOfClass, 1);
    end

    % Calculate the Within Class Scatter Matrix
    for i = 1:NumClasses
      samplesOfClass = Samples(Labels == Classes(i), :);
      Si = (1/classSamples(i)).*(transpose(samplesOfClass)*samplesOfClass);
      Sw = Sw + (P(i) * Si);
    end

    % Calculate the global mean
	  m0 = mean(mu);

    % Calculate the Between Class Scatter Matrix
    for i = 1:NumClasses
      Sb = Sb + P(i) * (transpose(mu(i,:) - m0) * (mu(i,:) - m0));
    endfor

    % Eigen matrix EigMat=inv(Sw)*Sb
    EigMat = inv(Sw)*Sb;

    % Perform Eigendecomposition
    [tmp_u, tmp_s] = eig(EigMat);
    [~,permu] = sort(diag(tmp_s), 'descend');
    tmp_s = tmp_s(permu,permu);
    tmp_u = tmp_u(:,permu);

    % Select the NewDim eigenvectors corresponding to the top NewDim
    % eigenvalues (Assuming they are NewDim<=NumClasses-1)
    %% You need to return the following variable correctly.
    A = zeros(NumFeatures,NewDim);  % Return the LDA projection vectors
    A = tmp_u(:, 1:NewDim);

function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% INIT
N = size(X, 1);
nfeatures = size(X, 2);
mu = zeros(1, nfeatures);
sigma = zeros(1, nfeatures);
X_norm = zeros(N, nfeatures);

for i = 1:nfeatures
  mu(i) = sum(X(:,i))/N; % mean of each column (feature)
end

for i = 1:nfeatures
  sigma(i) = sqrt(sum(((X(:,i) - mu(i)).^2))/(N - 1)); % standart deviation of each column
end

for i = 1:nfeatures
  for l = 1:N
    X_norm(l, i) = (X(l, i) - mu(i))/sigma(i); % normalize each column independently
  end
end

% ============================================================

end

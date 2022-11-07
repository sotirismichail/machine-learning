close all;
clear;
clc;

%% compute_size
%% Finds the size of the non-zero, ie, the detected digit's area
%% in the original image
function [begn, fini] = compute_size(img, s)
  % side == 1 == image height
  if s == 1
      img = img';
  end

  [rows, cols] = size(img);
  begn  = cols;
  fin = 1;

  for  i = 1 : rows
      for j = 1 : cols
          if(img(i,j)~=0)
              begn = min([begn j]);
              break;
          end
      end
  end

  for i = 1 : rows
      for j = cols: -1 : 1
          if(img(i,j)~=0)
               fin = max([fin j]);
              break;
          end
      end
  end

  begn = begn -0.5;
  fin = fin  + 0.5;
  fini = fin - begn;
end

%% draw_digit
%% Function to draw a rectangle around the detected digit
function draw_digit(img)
    figure();
    colormap(gray);
    image(img);
    hold on;
    [rec_w, width] = compute_size(img, 0);
    [rec_h, height] = compute_size(img, 1);
    plot([rec_w, rec_w],[rec_h, rec_h+height], 'r', 'LineWidth' , 4);
    plot([rec_w,rec_w+width ],[rec_h+height,rec_h+height],'r', 'LineWidth' , 4);
    plot([rec_w+width,rec_w+width],[rec_h+height,rec_h], 'r', 'LineWidth' , 4);
    plot([rec_w+width, rec_w], [rec_h,rec_h], 'r', 'LineWidth' , 4);

    hold off;
end

data_file = './data/mnist.mat';

data = load(data_file);

% Read the train data
[train_C1_indices, train_C2_indices,train_C1_images,train_C2_images] = read_data(data.trainX,data.trainY.');

% Read the test data
[test_C1_indices, test_C2_indices,test_C1_images,test_C2_images] = read_data(data.testX,data.testY.');

img_total_train = cat(1,train_C1_images,train_C2_images);
img_total_test = cat(1,test_C1_images,test_C2_images);

img_c1 = squeeze(train_C1_images(10,:,:));
img_c2 = squeeze(train_C2_images(10,:,:));
draw_digit(img_c1);
draw_digit(img_c2);
pause;


% Storing aspect ratios into a vector
aspect_ratio_train = [];
aspect_ratio_test = [];
for i = 1 : size(img_total_train,1)
   img_training = squeeze(img_total_train(i,:,:));
   aspect_ratio_train = [aspect_ratio_train computeAspectRatio(img_training)];
end

for i = 1 : size(img_total_test,1)
   img_testing = squeeze(img_total_test(i,:,:));
   aspect_ratio_test = [aspect_ratio_test computeAspectRatio(img_testing)];
end

% Keeping the aspect ratio maxima and minima
[aspect_ratio_min, idx_min] = min(aspect_ratio_train);
[aspect_ratio_max, idx_max] = max(aspect_ratio_train);

%% Bayesian Classifier

% A priori probabilities
samples_total = size(img_total_train,1);
samples_c1 = size(train_C1_images,1);
samples_c2 = size(train_C2_images,1);

p_c1 = samples_c1/samples_total
p_c2 = samples_c2/samples_total
% Class aspect ratio
mu = [mean(aspect_ratio_train(1:samples_c1)) mean(aspect_ratio_train(samples_c1+1:end))];

% Standard deviation of class aspect ratio
sigma = [sqrt(mean((aspect_ratio_train(1:samples_c1)-mu(1)).^2))...
  sqrt(mean((aspect_ratio_train(samples_c1+1:end)-mu(2)).^2)) ];

% Probabilities for a given C event
p_forc1 = normpdf(aspect_ratio_test, mu(1), sigma(1));
p_forc2 = normpdf(aspect_ratio_test, mu(2), sigma(2));

% A posteriori probabilities
p_postc1 = p_c1 * p_forc1;
p_postc2 = p_c2 * p_forc2;

% Classification result
false_pred = 0;

for i = 1 : size(img_total_test, 1)
    if p_postc1(i) >= p_postc2(i)
        bayes_class(i) = 1;
    else
        bayes_class(i) = 2;
    end
end

c1_pred = bayes_class(1:size(test_C1_images, 1));
c2_pred = bayes_class(size(test_C1_images,1)+1:end);

c1_false = sum(c1_pred == 2);
c2_false = sum(c2_pred == 1);

% Count misclassified objects
count_errors = c1_false + c2_false;

% Total Classification Error (%)
Error = 100*(count_errors/size(img_total_test, 1))


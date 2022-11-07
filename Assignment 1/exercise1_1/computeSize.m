 function [start, diff] = computeSize(img, s)
  % s == 1 == image height
  if s == 1
      img = img';
  end

  [rows, cols] = size(img);
  start  = cols;
  fin = 1;

  for  i = 1 : rows
      for j = 1 : cols
          if(img(i,j)~=0)
              start = min([start j]);
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

  start = start -0.5;
  fin = fin  + 0.5;
  diff = fin - start;
end

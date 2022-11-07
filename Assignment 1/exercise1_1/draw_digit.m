function draw_digit(fig, img)
  figure(fig);
  colormap(gray);
  image(img);
  hold on;
  [~, start_width, start_height, width, height] = computeAspectRatio(img);
  plot([start_width, start_width],[start_height, start_height+height], 'r', 'LineWidth' , 3);
  plot([start_width,start_width+width ],[start_height+height,start_height+height],'r', 'LineWidth' , 3 );
  plot([start_width+width,start_width+width],[start_height+height,start_height], 'r', 'LineWidth' , 3);
  plot([start_width+width, start_width], [start_height,start_height], 'r', 'LineWidth' , 3);
  hold off;
end

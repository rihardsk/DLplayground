function displayWrong(images, labels, pred)
  
  wronginds = find(labels != pred);
  m = size(wronginds, 1);
	grid = 5;
  disp = min(grid^2, m);
	randinds = randperm(m)(1:disp);
	selectedinds = wronginds(randinds);
  
	selected = images'(selectedinds, :);
	selectedpred = pred(selectedinds);
  selectedlabels = labels(selectedinds);
	
	figure;
	for i = 1:disp
		subplot(grid, grid, i);
		img = reshape(selected(i, :), 28, 28);
		imshow(img);
		title([num2str(selectedpred(i)), " (", num2str(selectedlabels(i)), ")"]);
	end
end

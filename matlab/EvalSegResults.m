seg_root = '/media/Work_HD/cxliu/datasets/VOCdevkit/VOC2012/';
seg_res_dir = '../example/val';
add_colormap = 0;
evaluate = 1;

%% add colormap to prediction image
if add_colormap
    load('pascal_seg_colormap.mat');
    imgs_dir = dir(fullfile(seg_res_dir, '*.png'));
    for i = 1:numel(imgs_dir)
        fprintf(1, 'adding colormap %d (%d) ...\n', i, numel(imgs_dir));
        img = imread(fullfile(seg_res_dir, imgs_dir(i).name));
        imwrite(img, colormap, fullfile(seg_res_dir, imgs_dir(i).name));
    end
end

%% evaluate IOU
if evaluate
    VOCopts = GetVOCopts(seg_root, seg_res_dir, 'train', 'val', 'VOC2012');
    [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts);
end
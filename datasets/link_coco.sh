mkdir coco
path_to_coco_dataset=$HOME/Data/coco
ln -s /$path_to_coco_dataset/annotations coco/annotations
ln -s /$path_to_coco_dataset/train2014 coco/train2014
ln -s /$path_to_coco_dataset/test2014 coco/test2014
ln -s /$path_to_coco_dataset/val2014 coco/val2014
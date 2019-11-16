#python train.py --name spade_zz_try1_noins --dataset_mode coco --dataroot ./datasets/coco_stuff_zz/ --no_instance --continue_train --tf_log
#python train.py --name spade_zz_try1_noins_bn --dataset_mode coco --dataroot ./datasets/coco_stuff_zz/ --no_instance --tf_log --norm_D spectralbatch --norm_E spectralbatch
#python train.py --name spade_zz_try1_noins_no_ganFeat_loss --dataset_mode coco --dataroot ./datasets/coco_stuff_zz/ --tf_log --no_instance --no_ganFeat_loss
python train.py --name spade_zz_try1_noins_single_scale_image --dataset_mode coco --dataroot ./datasets/coco_stuff_zz/ --tf_log --no_instance --netD multiscale

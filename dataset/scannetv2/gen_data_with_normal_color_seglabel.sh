#!/usr/bin/env bash
python prepare_data_inst.py --calc_normal --calc_color_seg_label  --calc_normal_seg_laebl --data_split train_normal_seg
python prepare_data_inst.py --calc_normal --calc_color_seg_label  --calc_normal_seg_laebl --data_split val_normal_seg

python prepare_data_inst_gttxt.py  --data_split val
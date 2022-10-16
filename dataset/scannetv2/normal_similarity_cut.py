# encoding: utf-8
"""
@author: Xiaodong Wu
@version: 1.0
@file: normal_similarity_cut.py

"""
import json
import numpy as np


class NormalSegLabelGenerator():
    def __init__(self):
        self.global_seg_id = 0

    def get_seg_label(self, seg_file_path):
        with open(seg_file_path) as f:
            overseg_js = json.load(f)

        seg_inds = np.asarray(overseg_js['segIndices'])

        unique_seg_inds = list(np.unique(seg_inds).astype(np.int32))
        label_mapper = {}
        for idx, seg_ind in enumerate(unique_seg_inds):
            label_mapper[seg_ind] = idx + self.global_seg_id

        normal_seg_labels = np.zeros(len(seg_inds)) - 1  # discarded label is -1

        for idx, seg_ind in enumerate(list(seg_inds)):
            normal_seg_labels[idx] = label_mapper[seg_ind]

        self.global_seg_id = self.global_seg_id + len(unique_seg_inds)

        return normal_seg_labels

if __name__ == '__main__':
    seg_file_path = 'data/scannet/scans/scene0111_00/scene0111_00_vh_clean_2.0.010000.segs.json'
    normal_seg_generator = NormalSegLabelGenerator()
    records = []
    for i in range(5):
        labels = normal_seg_generator.get_seg_label(seg_file_path)
        records.append(labels)


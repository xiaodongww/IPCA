# from  https://github.com/ScanNet/ScanNet/edit/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_label.py
# python imports
import math
import os, sys, argparse
import numpy as np
from sklearn import metrics

class SemIouEvaluator:
    def __init__(self, dataset='scannetv2'):
        if dataset == 'scannetv2':
            self.CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
        elif dataset == '3rscan':
            self.CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'counter',
                            'shelf', 'curtain', 'pillow', 'clothes', 'ceiling', 'fridge', 'tv', 'towel', 'plant', 'box',
                            'nightstand', 'toilet', 'sink', 'lamp', 'bathtub', 'object', 'blanket']

        self.num_cls = len(self.CLASS_LABELS)
        self.VALID_CLASS_IDS = np.arange(self.num_cls)
        self.UNKNOWN_ID = np.max(self.VALID_CLASS_IDS) + 1
        max_id = self.UNKNOWN_ID
        self.confusion = np.zeros((self.num_cls, self.num_cls), dtype=np.ulonglong)


    def evaluate_scan(self, pred_ids: np.array, gt_ids:np.array):
        # import ipdb
        # ipdb.set_trace()
        scan_confusion = metrics.confusion_matrix(y_true=gt_ids, y_pred=pred_ids, labels=np.arange(self.num_cls))
        self.confusion = self.confusion + scan_confusion

    def get_iou(self, label_id):
        if not label_id in self.VALID_CLASS_IDS:
            return float('nan')
        # #true positives
        tp = np.longlong(self.confusion[label_id, label_id])
        # #false negatives
        fn = np.longlong(self.confusion[label_id, :].sum()) - tp
        # #false positives
        not_ignored = [l for l in self.VALID_CLASS_IDS if not l == label_id]
        fp = np.longlong(self.confusion[not_ignored, label_id].sum())

        denom = (tp + fp + fn)
        if denom == 0:
            return float('nan')
        return (float(tp) / denom, tp, denom)


    def write_result_file(self, ious, filename):
        with open(filename, 'w') as f:
            f.write('iou scores\n')
            for i in range(len(self.VALID_CLASS_IDS)):
                label_id = self.VALID_CLASS_IDS[i]
                label_name = self.CLASS_LABELS[i]
                iou = ious[label_name][0]
                f.write('{0:<14s}({1:<2d}): {2:>5.3f}\n'.format(label_name, label_id, iou))
            f.write('\nconfusion matrix\n')
            f.write('\t\t\t')
            for i in range(len(self.VALID_CLASS_IDS)):
                #f.write('\t{0:<14s}({1:<2d})'.format(CLASS_LABELS[i], VALID_CLASS_IDS[i]))
                f.write('{0:<8d}'.format(self.VALID_CLASS_IDS[i]))
            f.write('\n')
            for r in range(len(self.VALID_CLASS_IDS)):
                f.write('{0:<14s}({1:<2d})'.format(self.CLASS_LABELS[r], self.VALID_CLASS_IDS[r]))
                for c in range(len(self.VALID_CLASS_IDS)):
                    f.write('\t{0:>5.3f}'.format(self.confusion[self.VALID_CLASS_IDS[r],self.VALID_CLASS_IDS[c]]))
                f.write('\n')
        print('wrote results to', filename)


    def evaluate(self):
        class_ious = {}
        for i in range(len(self.VALID_CLASS_IDS)):
            label_name = self.CLASS_LABELS[i]
            label_id = self.VALID_CLASS_IDS[i]
            class_ious[label_name] = self.get_iou(label_id)

        ious = []
        print('classes          IoU')
        print('----------------------------')
        for i in range(len(self.VALID_CLASS_IDS)):
            label_name = self.CLASS_LABELS[i]
            #print('{{0:<14s}: 1:>5.3f}'.format(label_name, class_ious[label_name][0]))
            if isinstance(class_ious[label_name], float):
                print('{0:<14s}: nan'.format(label_name))
            else:
                print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_ious[label_name][0], class_ious[label_name][1], class_ious[label_name][2]))
                ious.append(class_ious[label_name][0])
        print('mIoU = ', np.mean(ious))




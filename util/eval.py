# Modified from ScanNet evaluation script: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py

import os, sys, numpy as np
import util.utils_3d as util_3d
import util.utils as util

CLASS_LABELS = []
VALID_CLASS_IDS = np.array([])
ID_TO_LABEL = {}
LABEL_TO_ID = {}


def use_3rscan():
    global CLASS_LABELS
    global VALID_CLASS_IDS
    global ID_TO_LABEL
    global LABEL_TO_ID

    VALID_CLASS_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                       27]  # start from 1,  0 for annotated
    CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'counter',
                    'shelf', 'curtain', 'pillow', 'clothes', 'fridge', 'tv', 'towel', 'plant', 'box',
                    'nightstand', 'toilet', 'sink', 'lamp', 'bathtub', 'object', 'blanket']
    ID_TO_LABEL = {}
    LABEL_TO_ID = {}
    for i in range(len(VALID_CLASS_IDS)):
        LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
        ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

def use_scannet():
    global CLASS_LABELS
    global VALID_CLASS_IDS
    global ID_TO_LABEL
    global LABEL_TO_ID

    CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
                    'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
    VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
    ID_TO_LABEL = {}
    LABEL_TO_ID = {}
    for i in range(len(VALID_CLASS_IDS)):
        LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
        ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]


# ---------- Evaluation params ---------- #
# overlaps for evaluation
OVERLAPS             = np.append(np.arange(0.5,0.95,0.05), 0.25)
OVERLAPS = np.append(OVERLAPS, np.arange(0.05, 0.5, 0.05))
# minimum region size for evaluation [verts]
MIN_REGION_SIZES     = np.array( [ 100 ] )
# distance thresholds [m]
DISTANCE_THRESHES    = np.array( [  float('inf') ] )
# distance confidences
DISTANCE_CONFS       = np.array( [ -float('inf') ] )


def evaluate_matches(matches, dataset):
    if dataset == 'scannetv2':
        use_scannet()
    elif dataset == '3rscan':
        use_3rscan()
    else:
        raise KeyError
    overlaps = OVERLAPS
    min_region_sizes = [MIN_REGION_SIZES[0]]
    dist_threshes = [DISTANCE_THRESHES[0]]
    dist_confs = [DISTANCE_CONFS[0]]
    # results: class x overlap
    ap = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), np.float)
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]['pred']:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]['pred'][label_name]:
                            if 'filename' in p:
                                pred_visited[p['filename']] = False
            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]['pred'][label_name]
                    gt_instances = matches[m]['gt'][label_name]
                    # filter groups in ground truth
                    gt_instances = [gt for gt in gt_instances if
                                    gt['instance_id'] >= 1000 and gt['vert_count'] >= min_region_size and gt['med_dist'] <= distance_thresh and gt['dist_conf'] >= distance_conf]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=np.bool)
                    # collect matches
                    for (gti, gt) in enumerate(gt_instances):
                        found_match = False
                        num_pred = len(gt['matched_pred'])
                        for pred in gt['matched_pred']:
                            # greedy assignments
                            if pred_visited[pred['filename']]:
                                continue
                            overlap = float(pred['intersection']) / (
                            gt['vert_count'] + pred['vert_count'] - pred['intersection'])
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['filename']] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred['matched_gt']:
                            overlap = float(gt['intersection']) / (
                            gt['vert_count'] + pred['vert_count'] - gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                # group?
                                if gt['instance_id'] < 1000:
                                    num_ignore += gt['intersection']
                                # small ground truth instances
                                if gt['vert_count'] < min_region_size or gt['med_dist'] > distance_thresh or gt['dist_conf'] < distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(num_ignore) / pred['vert_count']
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    if(len(y_true_sorted_cumsum) == 0):
                        num_true_examples = 0
                    else:
                        num_true_examples = y_true_sorted_cumsum[-1]
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    # first point in curve is artificial
                    precision[-1] = 1.
                    recall[-1] = 0.

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)

                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float('nan')
                ap[di, li, oi] = ap_current
    return ap


def compute_averages(aps, dataset):
    if dataset == 'scannetv2':
        use_scannet()
    elif dataset == '3rscan':
        use_3rscan()
    else:
        raise KeyError
    d_inf = 0
    o50   = np.where(np.isclose(OVERLAPS,0.5))
    o25   = np.where(np.isclose(OVERLAPS,0.25))
    # oAllBut25  = np.where(np.logical_not(np.isclose(OVERLAPS,0.25)))
    oAllBut25  = np.where(OVERLAPS>=0.5)
    avg_dict = {}
    #avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,oAllBut25])
    avg_dict['all_ap_50%'] = np.nanmean(aps[ d_inf,:,o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[ d_inf,:,o25])
    avg_dict["classes"]  = {}
    for (li,label_name) in enumerate(CLASS_LABELS):
        avg_dict["classes"][label_name]             = {}
        #avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
        avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,oAllBut25])
        avg_dict["classes"][label_name]["ap50%"]    = np.average(aps[ d_inf,li,o50])
        avg_dict["classes"][label_name]["ap25%"]    = np.average(aps[ d_inf,li,o25])
    return avg_dict


def assign_instances_for_scan(scene_name, pred_info, gt_file, dataset):
    if dataset == 'scannetv2':
        use_scannet()
    elif dataset == '3rscan':
        use_3rscan()
    else:
        raise KeyError
    try:
        gt_ids = util_3d.load_ids(gt_file)
    except Exception as e:
        util.print_error('unable to load ' + gt_file + ': ' + str(e))

    # get gt instances
    gt_instances = util_3d.get_instances(gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)
    # associate
    gt2pred = gt_instances.copy()
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt['matched_pred'] = []
    pred2gt = {}
    for label in CLASS_LABELS:
        pred2gt[label] = []
    num_pred_instances = 0
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.in1d(gt_ids//1000, VALID_CLASS_IDS))
    # go thru all prediction masks
    nMask = pred_info['label_id'].shape[0]
    for i in range(nMask):
        label_id = int(pred_info['label_id'][i])
        conf = pred_info['conf'][i]
        if not label_id in ID_TO_LABEL:
            continue
        label_name = ID_TO_LABEL[label_id]
        # read the mask
        pred_mask = pred_info['mask'][i]   # (N), long
        if len(pred_mask) != len(gt_ids):
            util.print_error('wrong number of lines in mask#%d: ' % (i)  + '(%d) vs #mesh vertices (%d)' % (len(pred_mask), len(gt_ids)))
        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < MIN_REGION_SIZES[0]:
            continue  # skip if empty

        pred_instance = {}
        pred_instance['filename'] = '{}_{:03d}'.format(scene_name, num_pred_instances)
        pred_instance['pred_id'] = num_pred_instances
        pred_instance['label_id'] = label_id
        pred_instance['vert_count'] = num
        pred_instance['confidence'] = conf
        pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))

        # matched gt instances
        matched_gt = []
        # go thru all gt instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy['intersection']   = intersection
                pred_copy['intersection'] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
        pred_instance['matched_gt'] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)
    return gt2pred, pred2gt


def print_results(avgs, logger, dataset):
    if dataset == 'scannetv2':
        use_scannet()
    elif dataset == '3rscan':
        use_3rscan()
    else:
        raise KeyError
    sep     = ""
    col1    = ":"
    lineLen = 64

    if logger is None:
        print("")
        print("#" * lineLen)
    else:
        logger.info("")
        logger.info("#" * lineLen)
    line  = ""
    line += "{:<15}".format("what"      ) + sep + col1
    line += "{:>15}".format("AP"        ) + sep
    line += "{:>15}".format("AP_50%"    ) + sep
    line += "{:>15}".format("AP_25%"    ) + sep
    if logger is None:
        print(line)
        print("#" * lineLen)
    else:
        logger.info(line)
        logger.info("#" * lineLen)

    for (li,label_name) in enumerate(CLASS_LABELS):
        ap_avg  = avgs["classes"][label_name]["ap"]
        ap_50o  = avgs["classes"][label_name]["ap50%"]
        ap_25o  = avgs["classes"][label_name]["ap25%"]
        line  = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_avg ) + sep
        line += sep + "{:>15.3f}".format(ap_50o ) + sep
        line += sep + "{:>15.3f}".format(ap_25o ) + sep
        if logger is None:
            print(line)
        else:
            logger.info(line)

    all_ap_avg  = avgs["all_ap"]
    all_ap_50o  = avgs["all_ap_50%"]
    all_ap_25o  = avgs["all_ap_25%"]

    if logger is None:
        print("-" * lineLen)
    else:
        logger.info("-" * lineLen)
    line  = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(all_ap_avg)  + sep
    line += "{:>15.3f}".format(all_ap_50o)  + sep
    line += "{:>15.3f}".format(all_ap_25o)  + sep
    if logger is None:
        print(line)
        print("")
    else:
        logger.info(line)
        logger.info("")

def write_csv_result(avgs, result_path, dataset):
    if dataset == 'scannetv2':
        use_scannet()
    elif dataset == '3rscan':
        use_3rscan()
    else:
        raise KeyError
    base_dir = os.path.basename(result_path)
    os.makedirs(base_dir, exist_ok=True)
    line0 = ","
    line1 = "AP,"
    line2 = "AP@25,"
    line3 = "AP@50,"

    all_ap_avg  = avgs["all_ap"]
    all_ap_50o  = avgs["all_ap_50%"]
    all_ap_25o  = avgs["all_ap_25%"]

    for (li, label_name) in enumerate(CLASS_LABELS):
        ap_avg = avgs["classes"][label_name]["ap"]
        ap_50o = avgs["classes"][label_name]["ap50%"]
        ap_25o = avgs["classes"][label_name]["ap25%"]

        line0 += "{},".format(label_name)
        line1 += "{:>15.3f},".format(ap_avg)
        line2 += "{:>15.3f},".format(ap_50o)
        line3 += "{:>15.3f},".format(ap_25o)
    line0 += "total"
    line1 += "{:>15.3f},".format(all_ap_avg)
    line2 += "{:>15.3f},".format(all_ap_50o)
    line3 += "{:>15.3f},".format(all_ap_25o)


    with open(result_path, 'w')  as f:
        f.write("{}\n".format(line0))
        f.write("{}\n".format(line1))
        f.write("{}\n".format(line2))
        f.write("{}\n".format(line3))

def evaluate_prec_recall(matches, score_thresh, iou_thresh, dataset):
    if dataset == 'scannetv2':
        use_scannet()
    elif dataset == '3rscan':
        use_3rscan()
    else:
        raise KeyError

    precs = []
    recalls = []
    for li, label_name in enumerate(CLASS_LABELS):
        pred_labels = []
        gth_recall_labels = []
        for scene_name in matches.keys():
            gt = matches[scene_name]['gt'][label_name]
            pred = [candidate for candidate in matches[scene_name]['pred'][label_name] if candidate['confidence'] >= score_thresh]

            for obj in gt:
                is_match = False
                gt_vert_count = obj['vert_count']
                for candidate in obj['matched_pred']:
                    overlap = float(candidate['intersection']) / (
                        gt_vert_count + candidate['vert_count'] - candidate['intersection'])

                    if overlap >= iou_thresh:
                        is_match = True
                gth_recall_labels.append(is_match)

            for candidate in pred:
                is_match = False
                pred_vert_count = candidate['vert_count']
                for obj in candidate['matched_gt']:
                    overlap = float(obj['intersection']) / (
                        pred_vert_count + obj['vert_count'] - obj['intersection'])

                    if overlap >= iou_thresh:
                        is_match = True
                pred_labels.append(is_match)
        prec = np.array(gth_recall_labels).sum() / (len(pred_labels) + 1e-6)
        recall = np.array(gth_recall_labels).sum() / (len(gth_recall_labels) + 1e-6)

        precs.append(prec)
        recalls.append(recall)

    for label_name, prec, recall in zip(CLASS_LABELS, precs, recalls):
        print('{}, prec={:.3f}, recall={:.3f}'.format(label_name, prec, recall))


    return np.mean(precs), np.mean(recalls)
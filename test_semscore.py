'''
test_semscore.py
Scripts for testing models with SCN module.
'''

import torch
import time
import numpy as np
import random
import os, glob
import copy

import util.utils as utils
import util.eval as eval
from util.evaluate_semantic_label import SemIouEvaluator


def init():
    global cfg
    from util.config import get_parser
    cfg = get_parser()
    cfg.task = 'test'
    cfg.dist = False

    global sem_evaluator
    sem_evaluator = SemIouEvaluator(dataset=cfg.dataset)
    global result_dir
    result_dir = os.path.join(cfg.exp_path, 'result',
                              'epoch{}_nmst{}_scoret{}_npointt{}'.format(cfg.test_epoch, cfg.TEST_NMS_THRESH,
                                                                         cfg.TEST_SCORE_THRESH, cfg.TEST_NPOINT_THRESH),
                              cfg.split)
    backup_dir = os.path.join(result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'predicted_masks'), exist_ok=True)
    os.system('cp test.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    global semantic_label_idx

    if cfg.dataset == 'scannetv2':
        semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    elif cfg.dataset == '3rscan':
        semantic_label_idx = list(range(1, 28))

    global logger
    from util.log import get_logger
    logger = get_logger(cfg)

    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)


def correct_pred_by_overseg_label(predictions: np.array, oversegment_labels: np.array, ignore_classes: list = []):
    """
    :param predictions: shape (N,) each element is the class label
    :param oversegment_labels:  shape (N,) each element is the over-segment index the vertex belongs to
    :return: np.array shape(N)
    """
    corrected_pred = copy.copy(predictions)
    uni_seg_labels = np.unique(oversegment_labels).astype(np.int32)
    uni_seg_labels = uni_seg_labels[uni_seg_labels >= 0]

    for overseg_label in uni_seg_labels:
        mask = oversegment_labels == overseg_label
        if np.sum(mask) < 25:
            continue
        pred_ = predictions[mask]
        most_frequent_label = Counter(pred_).most_common(1)[0][0]
        if most_frequent_label not in ignore_classes:
            corrected_pred[mask] = most_frequent_label

        return corrected_pred


def correct_inst_by_overseg_label(clusters: torch.Tensor, oversegment_labels: torch.Tensor,
                                  correct_ratio_thresh: float = 0.3):
    """
    For each segment, if more than 60% points belong to an instance proposal, then make all points
    :param clusters:
    :param oversegment_labels:
    :return:
    """
    num_corrected = 0

    for segid in torch.unique(oversegment_labels):
        seg_mask = oversegment_labels == segid
        if torch.sum(seg_mask) <= 1:
            continue

        seg_size = torch.sum(seg_mask)
        seg_proposal_labels = clusters[:, seg_mask]  # binary tensor with shape (num_proposals, seg_size)
        largest_cluster_size = seg_proposal_labels.sum(1).max()
        largest_cluster_id = torch.where(seg_proposal_labels.sum(1) == largest_cluster_size)[0]


        ratio = largest_cluster_size / seg_size
        if ratio > correct_ratio_thresh:  # 0.6 AP=0.395, 0.3 AP=0.399, 0.1
            clusters[np.ix_(largest_cluster_id.cpu().numpy(), torch.where(seg_mask)[0].cpu().numpy())] = 1
            largest_cluster_id.unsqueeze(1)

            num_corrected += (seg_size - largest_cluster_size)
    print('[Cluster Correction] {} points corrected'.format(num_corrected))
    return clusters



def evaluate_semantic_segmantation_accuracy(matches):
    seg_gt_list = []
    seg_pred_list = []
    for k, v in matches.items():
        seg_gt_list.append(v['seg_gt'])
        seg_pred_list.append(v['seg_pred'])
    seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
    seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
    assert seg_gt_all.shape == seg_pred_all.shape
    correct = (seg_gt_all[seg_gt_all != -100] == seg_pred_all[seg_gt_all != -100]).sum()
    whole = (seg_gt_all != -100).sum()
    seg_accuracy = correct.float() / whole.float()
    return seg_accuracy

def evaluate_semantic_segmantation_miou(matches):
    seg_gt_list = []
    seg_pred_list = []
    for k, v in matches.items():
        seg_gt_list.append(v['seg_gt'])
        seg_pred_list.append(v['seg_pred'])
    seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
    seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
    assert seg_gt_all.shape == seg_pred_all.shape
    iou_list = []
    for _index in seg_gt_all.unique():
        if _index != -100:
            intersection = ((seg_gt_all == _index) &  (seg_pred_all == _index)).sum()
            union = ((seg_gt_all == _index) | (seg_pred_all == _index)).sum()
            iou = intersection.float() / union
            iou_list.append(iou)
    iou_tensor = torch.tensor(iou_list)
    miou = iou_tensor.mean()
    return miou



def test(model, model_fn, dataset, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    with torch.no_grad():
        model = model.eval()
        start = time.time()

        matches = {}
        cls_accs = []
        correct_earnings = []
        total_forward_time = 0
        total_geninst_time = 0
        for i, batch in enumerate(dataset.test_data_loader):
            N = batch['feats'].shape[0]
            if cfg.dataset == 'scannetv2':
                test_scene_name = dataset.test_file_names[int(batch['id'][0])].split('/')[-1][:12]
                print(batch['scene_names'])

            elif cfg.dataset == '3rscan':
                test_scene_name = dataset.test_file_names[int(batch['id'][0])].split('/')[-1].replace('.pth', '')

            start1 = time.time()
            preds = model_fn(batch, model, epoch)
            end1 = time.time() - start1
            t_forward = end1
            total_forward_time += end1


            ##### get predictions (#1 semantic_pred, pt_offsets; #2 scores, proposals_pred)
            semantic_scores = preds['semantic']  # (N, nClass=20) float32, cuda
            semantic_pred = semantic_scores.max(1)[1]  # (N) long, cuda
            pt_offsets = preds['pt_offsets']  # (N, 3), float32, cuda
            # print('semantic classes: ', torch.unique(semantic_pred))

            if cfg.test_upper_bound:
                semantic_pred = batch['labels']

            normal_seg_labels = batch['normal_seg_labels']
            if cfg.cluster_implicit_parts:
                semantic_pred_arr = semantic_pred.cpu().numpy()
                # wall and floor are not corrected, only objects' points are taken into account. set ignore_classes=[0,1]
                semantic_pred_arr = utils.correct_pred_by_overseg_label(semantic_pred_arr,
                                                                        np.array(normal_seg_labels).astype(
                                                                            np.int32), ignore_classes=[0, 1])
                semantic_pred = torch.Tensor(semantic_pred_arr).cuda().long()

            if cfg.split == 'val':
                semantic_pred_arr = semantic_pred.cpu().numpy()
                semantic_gth_arr = batch['labels'].cpu().numpy()  # (N) long, cuda
                # exclude 'unlabeled', 'wall', 'floor' points
                valid_mask = semantic_gth_arr != -100
                valid_mask = np.logical_and(valid_mask, semantic_gth_arr != 0)
                valid_mask = np.logical_and(valid_mask, semantic_gth_arr != 1)


                correct = semantic_gth_arr[valid_mask] == semantic_pred_arr[valid_mask]
                acc = np.sum(correct) / float(len(correct))
                cls_accs.append(acc)
                semantic_pred = torch.Tensor(semantic_pred_arr).cuda().long()
                sem_evaluator.evaluate_scan(pred_ids=semantic_pred_arr, gt_ids=semantic_gth_arr)
                # print('semantic seg acc: {}'.format(acc))
            gen_inst_start = time.time()
            if (epoch > cfg.prepare_epochs):
                proposals_idx, proposals_offset = preds['proposals']

                ### use obj classification score multiply iou prediction as proposal score
                import torch.nn.functional as F
                obj_scores = F.softmax(preds['obj_scores'], 1)   # only use forground classes
                iou_scores_pred = torch.sigmoid(preds['score'].view(-1))
                scores_pred = obj_scores * iou_scores_pred.unsqueeze(1)

                if cfg.dataset == 'scannetv2':
                    scores_pred = scores_pred[:, 2:].T.contiguous().view(-1)

                    # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                    # proposals_offset: (nProposal + 1), int, cpu
                    proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int,
                                                 device=scores_pred.device)  # (nProposal, N), int, cuda
                    proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

                    proposals_pred = proposals_pred.repeat(18, 1)  # 18 foreground classes
                    semantic_id = torch.cat([torch.ones(len(proposals_offset) -1) * i for i in range(2, 20)])
                    semantic_id = torch.tensor(semantic_label_idx, device=scores_pred.device)[semantic_id.long()]
                elif cfg.dataset == '3rscan':
                    valid_cls_idx = torch.ones(27) == 1
                    valid_cls_idx[0] = False  #
                    valid_cls_idx[1] = False  #
                    valid_cls_idx[14] = False  # ceiling
                    scores_pred = scores_pred[:, valid_cls_idx].T.contiguous().view(-1)
                    proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int,
                                                 device=scores_pred.device)  # (nProposal, N), int, cuda
                    proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
                    proposals_pred = proposals_pred.repeat(24, 1)  # 24 foreground classes
                    semantic_id = torch.cat([torch.ones(len(proposals_offset) -1) * i for i in torch.where(valid_cls_idx)[0].tolist()])
                    semantic_id = torch.tensor(semantic_label_idx, device=scores_pred.device)[semantic_id.long()]


                ##### score threshold  (perf. better than two threshold)
                score_mask = (scores_pred > cfg.TEST_SCORE_THRESH)
                scores_pred = scores_pred[score_mask]
                proposals_pred = proposals_pred[score_mask]
                semantic_id = semantic_id[score_mask]


                ##### npoint threshold
                proposals_pointnum = proposals_pred.sum(1)
                npoint_mask = (proposals_pointnum > cfg.TEST_NPOINT_THRESH)
                scores_pred = scores_pred[npoint_mask]
                proposals_pred = proposals_pred[npoint_mask]
                semantic_id = semantic_id[npoint_mask]

                pick_idxs = np.ones(len(proposals_pred)) == 1
                clusters = proposals_pred[pick_idxs]
                cluster_scores = scores_pred[pick_idxs]
                cluster_semantic_id = semantic_id[pick_idxs]



                ### cluster implicit parts
                if cfg.cluster_implicit_parts:
                    clusters = correct_inst_by_overseg_label(clusters, oversegment_labels=normal_seg_labels)

                nclusters = clusters.shape[0]

                ##### prepare for evaluation
                if cfg.eval:
                    pred_info = {}
                    pred_info['conf'] = cluster_scores.cpu().numpy()
                    pred_info['label_id'] = cluster_semantic_id.cpu().numpy()
                    pred_info['mask'] = clusters.cpu().numpy()
                    gt_file = os.path.join(cfg.data_root, cfg.dataset, cfg.split + '_gt', test_scene_name + '.txt')
                    gt2pred, pred2gt = eval.assign_instances_for_scan(test_scene_name, pred_info, gt_file, dataset=cfg.dataset)
                    matches[test_scene_name] = {}
                    matches[test_scene_name]['gt'] = gt2pred
                    matches[test_scene_name]['pred'] = pred2gt
                    if cfg.split == 'val':
                        matches[test_scene_name]['seg_gt'] = batch['labels']
                        matches[test_scene_name]['seg_pred'] = semantic_pred
            gen_inst_stop = time.time()
            t_gen_inst = gen_inst_stop - gen_inst_start
            total_geninst_time += t_gen_inst
            print('{} / {}, t_forward: {}s,  t_gen_inst: {}s, avg_fwd: {}s, avg_inst: {}s '.format(i+1, len(dataset.test_data_loader), t_forward, t_gen_inst,
                                                                                                   total_forward_time / (i+1), total_geninst_time / (i+1)))

            ##### save files
            start3 = time.time()
            if cfg.save_semantic:
                os.makedirs(os.path.join(result_dir, 'semantic'), exist_ok=True)
                semantic_np = semantic_pred.cpu().numpy()
                print('saved in {}'.format(os.path.join(result_dir, 'semantic', test_scene_name + '.npy')))
                np.save(os.path.join(result_dir, 'semantic', test_scene_name + '.npy'), semantic_np)

            if cfg.save_pt_offsets:
                os.makedirs(os.path.join(result_dir, 'coords_offsets'), exist_ok=True)
                pt_offsets_np = pt_offsets.cpu().numpy()
                coords_np = batch['locs_float'].numpy()
                coords_offsets = np.concatenate((coords_np, pt_offsets_np), 1)  # (N, 6)
                np.save(os.path.join(result_dir, 'coords_offsets', test_scene_name + '.npy'), coords_offsets)


            if (epoch > cfg.prepare_epochs and cfg.save_instance):
                f = open(os.path.join(result_dir, test_scene_name + '.txt'), 'w')
                print(test_scene_name)

                cluster_labels = cluster_semantic_id.cpu().numpy()
                for proposal_id in range(nclusters):
                    clusters_i = clusters[proposal_id].cpu().numpy()  # (N)
                    semantic_label = cluster_labels[proposal_id]
                    score = cluster_scores[proposal_id]
                    f.write('predicted_masks/{}_{:03d}.txt {} {:.4f}'.format(test_scene_name, proposal_id,
                                                                             semantic_label, score))
                    if proposal_id < nclusters - 1:
                        f.write('\n')

                    content = list(map(lambda x: str(x), clusters_i.tolist()))
                    content = "\n".join(content)
                    with open(
                            os.path.join(
                                result_dir, "predicted_masks",
                                test_scene_name + "_%03d.txt" % (proposal_id)),
                            "w") as cf:
                        cf.write(content)
                f.close()

            end3 = time.time() - start3
            end = time.time() - start
            start = time.time()
            if cfg.eval:
                sem_evaluator.evaluate()

            ##### print
            logger.info(
                "instance iter: {}/{} point_num: {} ncluster: {} time: total {:.2f}s inference {:.2f}s save {:.2f}s".format(
                    batch['id'][0] + 1, len(dataset.test_file_names), N, nclusters, end, end1, end3))

            if cfg.eval:
                logger.info("avg classification acc: {}".format(sum(cls_accs) / len(cls_accs)))
                logger.info("Total correcting earnings: {}".format(np.sum(correct_earnings)))
            torch.cuda.empty_cache()
            del proposals_pred

        # evaluate semantic segmantation accuracy and mIoU
        if cfg.split == 'val':
            seg_accuracy = evaluate_semantic_segmantation_accuracy(matches)
            logger.info("semantic_segmantation_accuracy: {:.4f}".format(seg_accuracy))
            miou = evaluate_semantic_segmantation_miou(matches)
            logger.info("semantic_segmantation_mIoU: {:.4f}".format(miou))


        ##### evaluation
        if cfg.eval:
            ap_scores = eval.evaluate_matches(matches, dataset=cfg.dataset)
            avgs = eval.compute_averages(ap_scores, dataset=cfg.dataset)
            eval.print_results(avgs, logger, dataset=cfg.dataset)

            ### print AP from 0.5 to 0.9 (note that ScanNet's AP do not contain AP@95)
            for thresh_ in np.arange(0.05, 0.95, 0.05)[::-1]:
                OVERLAPS = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
                OVERLAPS = np.append(OVERLAPS, np.arange(0.05, 0.5, 0.05))

                thresh_idx = np.where(np.isclose(OVERLAPS, thresh_))
                ap_ = np.nanmean(ap_scores[0, :, thresh_idx])
                print('AP@{} : {:.4f}'.format(int(thresh_ * 100), ap_))



            sem_evaluator.evaluate()
            try:
                eval_out_path = os.path.join(result_dir, 'evaluation_metric_result_TEST_NPOINT_THRESH_{}_{:.4f}.csv'.format(cfg.TEST_NPOINT_THRESH, cfg.TEST_SCORE_THRESH))
                eval.write_csv_result(avgs, eval_out_path, dataset=cfg.dataset)
                pass
            except:
                pass


def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


if __name__ == '__main__':
    init()

    ##### SA
    if cfg.cache:
        if cfg.dataset == 'scannetv2':
            test_file_names = sorted(
                glob.glob(os.path.join(cfg.data_root, cfg.dataset, cfg.test_dir, '*' + cfg.filename_suffix)))
            utils.create_shared_memory(test_file_names, wlabel=True)
        elif cfg.dataset == '3rscan':
            test_file_names = sorted(
                glob.glob(os.path.join(cfg.data_root, cfg.dataset, cfg.test_dir, '*' + cfg.filename_suffix)))
            utils.create_shared_memory_3rscan(test_file_names, wlabel=True)

    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    ##### model
    logger.info('=> creating model ...')
    logger.info('Classes: {}'.format(cfg.classes))

    if model_name == 'pointgroup':
        from model.pointgroup.pointgroup import PointGroup as Network
        from model.pointgroup.pointgroup import model_fn_decorator
    elif model_name == 'segspgsemleaky':
        from model.seggroup.segspgsemleaky import SegSPGSemLeaky as Network
        from model.seggroup.segspgsemleaky import model_fn_decorator
    elif model_name == 'segspgsemleakysemscore':
        from model.seggroup.segspgsemleakysemscore import SegSPGSemLeakySemScore as Network
        from model.seggroup.segspgsemleakysemscore import model_fn_decorator
    elif model_name == 'pointgroupleaky':
        from model.pointgroup.pointgroupleaky import PointGroupLeaky as Network
        from model.pointgroup.pointgroupleaky import model_fn_decorator
    else:
        print("Error: no model - " + model_name)
        exit(0)

    cfg.pretrain_path = cfg.pretrain  # cfg.pretrain_path is used for initialization of training process, overwrite it
    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    # logger.info(model)
    logger.info('#classifier parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### model_fn (criterion)
    model_fn = model_fn_decorator(cfg, test=True)

    ##### load model
    pretrain_path = cfg.pretrain
    logger.info('Restore from {}'.format(pretrain_path))

    gpu = 0
    map_location = {'cuda:0': 'cuda:{}'.format(gpu)} if gpu > 0 else None
    checkpoint = torch.load(pretrain_path, map_location=map_location)
    for k, v in checkpoint.items():
        if 'module.' in k:
            checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}
        break
    model.load_state_dict(checkpoint)

    ##### data
    if cfg.dataset == 'scannetv2':
        from data.scannetv2_inst import Dataset
        dataset = Dataset(cfg, test=True)
        dataset.testLoader()
        logger.info('Testing samples ({}): {}'.format(cfg.split, len(dataset.test_file_names)))
    elif cfg.dataset == '3rscan':
        import data.threerscan_inst
        dataset = data.threerscan_inst.Dataset(cfg, test=True)
        dataset.testLoader()
        logger.info('Testing samples ({}): {}'.format(cfg.split, len(dataset.test_file_names)))
    else:
        print("Error: no dataset - " + cfg.dataset)
        exit(0)


    ##### evaluate
    test(model, model_fn, dataset, cfg.test_epoch)

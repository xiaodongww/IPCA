# encoding: utf-8
"""
@author: Xiaodong Wu
@version: 1.0
@file: segspgsemleaky.py

Baseline model + IPCA module
"""

import torch
import torch.nn as nn
import spconv
import functools
import sys, os
import dgl
from dgl.nn.pytorch import SAGEConv
from dgl.transform import add_reverse_edges

sys.path.append('../../')

from lib.pointgroup_ops.functions import pointgroup_ops
from util import utils
from ..pointgroup.pointgroup import ResidualBlock, VGGBlock, UBlock
from .pointnet2_net import segmentPointNet2



class SegSAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, num_class, aggregator_type):
        super().__init__()
        self.conv1 = SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type=aggregator_type)
        self.conv2 = SAGEConv(
            in_feats=hid_feats, out_feats=hid_feats, aggregator_type=aggregator_type)
        self.score_layer = nn.Linear(hid_feats, num_class)
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, graph, inputs):
        # inputs are features of nodes
        graph = dgl.add_self_loop(graph)

        h = self.conv1(graph, inputs)
        h = self.activation(h)
        h = self.conv2(graph, h)

        score = self.score_layer(h)

        return score


class SegSPGSemLeaky(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        input_c = cfg.input_channel
        m = cfg.m
        classes = cfg.classes
        block_reps = cfg.block_reps
        block_residual = cfg.block_residual

        self.cluster_radius = cfg.cluster_radius
        self.cluster_meanActive = cfg.cluster_meanActive
        self.cluster_shift_meanActive = cfg.cluster_shift_meanActive
        self.cluster_npoint_thre = cfg.cluster_npoint_thre

        self.score_scale = cfg.score_scale
        self.score_fullscale = cfg.score_fullscale
        self.mode = cfg.score_mode

        self.prepare_epochs = cfg.prepare_epochs

        self.pretrain_path = cfg.pretrain_path
        self.pretrain_module = cfg.pretrain_module
        self.fix_module = cfg.fix_module

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        if cfg.use_coords:
            input_c += 3

        if 'use_normal' in cfg:
            if cfg.use_normal:
                input_c += 3

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )

        self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block, indice_key_id=1)

        self.output_layer = spconv.SparseSequential(
            norm_fn(m),
            nn.LeakyReLU(0.01)
        )

        #### semantic segmentation
        self.linear = nn.Linear(m, classes)  # bias(default): True

        #### offset
        self.offset = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.LeakyReLU(0.01)
        )
        self.offset_linear = nn.Linear(m, 3, bias=True)

        #### score branch
        self.score_unet = UBlock([m, 2 * m], norm_fn, 2, block, indice_key_id=1)
        self.score_outputlayer = spconv.SparseSequential(
            norm_fn(m),
            nn.LeakyReLU(0.01)
        )
        self.score_linear = nn.Linear(m, 1)

        self.apply(self.set_bn_init)

        #### segment feature learning branch
        use_bn = False
        self.segment_layer = segmentPointNet2(
            fea_dim=m + 3,
            out_fea_dim=self.cfg.segment_fea_dim,
            use_bn=use_bn)  # use global coordinate as fea. use local coord. as points' loc

        self.segsem_layer = SegSAGE(self.cfg.segment_fea_dim, self.cfg.segment_fea_dim, self.cfg.classes,
                                    self.cfg.graph_aggregator_type)

        #### fix parameter
        module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
                      'linear': self.linear, 'offset': self.offset, 'offset_linear': self.offset_linear,
                      'score_unet': self.score_unet, 'score_outputlayer': self.score_outputlayer,
                      'score_linear': self.score_linear, 'segment_layer': self.segment_layer,
                      'segsem_layer': self.segsem_layer}

        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

        #### load pretrain weights
        if self.pretrain_path is not None:
            map_location = {'cuda:0': 'cuda:{}'.format(cfg.local_rank)} if cfg.local_rank > 0 else None
            pretrain_dict = torch.load(self.pretrain_path, map_location=map_location)
            if 'module.' in list(pretrain_dict.keys())[0]:
                pretrain_dict = {k[len('module.'):]: v for k, v in pretrain_dict.items()}
            for m in self.pretrain_module:
                print('Loading {} module from {}'.format(m, self.pretrain_path))
                n1, n2 = utils.load_model_param(module_map[m], pretrain_dict, prefix=m)
                if cfg.local_rank == 0:
                    print("[PID {}] Load pretrained ".format(os.getpid()) + m + ": {}/{}".format(n1, n2))

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, fullscale, scale, mode):
        '''
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        '''
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean = pointgroup_ops.sec_mean(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_mean = torch.index_select(clusters_coords_mean, 0,
                                                  clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = pointgroup_ops.sec_min(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_max = pointgroup_ops.sec_max(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float

        clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[
            0] - 0.01  # (nCluster), float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = - min_xyz + torch.clamp(fullscale - range - 0.001, min=0) * torch.rand(3).cuda() + torch.clamp(
            fullscale - range + 0.001, max=0) * torch.rand(3).cuda()
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()],
                                    1)  # (sumNPoint, 1 + 3)

        out_coords, inp_map, out_map = pointgroup_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1,
                                                                       mode)
        # output_coords: M * (1 + 3) long
        # input_map: sumNPoint int
        # output_map: M * (maxActive + 1) int

        out_feats = pointgroup_ops.voxelization(clusters_feats, out_map.cuda(), mode)  # (M, C), float, cuda

        spatial_shape = [fullscale] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats, out_coords.int().cuda(), spatial_shape,
                                                     int(clusters_idx[-1, 0]) + 1)

        return voxelization_feats, inp_map

    def forward(self, input, input_map, coords, batch_idxs, batch_offsets, spg_edgs, spg_edg_offsets, seg_pcd_idxs,
                uni_seg_labels, spg_edg_idxs, epoch, other_input={}):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        :param spg_edgs: # shape: (num_batch_segs, 2) each line is an edge containing src_seg_id and tgt_seg_id
        :param spg_edg_offsets:   # shape: (B + 1)
        :param other_input: dict, used for calc instance seg. upper bound by input gth semantic segmentation
         (other_input['gth_sem_seg_label']) or boost performace by input over-segment label (other_input['over_seg_label']).
        '''

        ret = {}
        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]

        #### offset
        pt_offsets_feats = self.offset(output_feats)
        pt_offsets = self.offset_linear(pt_offsets_feats)  # (N, 3), float32
        ret['pt_offsets'] = pt_offsets

        #### semantic segmentation
        semantic_scores = self.linear(output_feats)  # (N, nClass), float
        sampled_segment_coords = coords[seg_pcd_idxs]  # shape (num_segs, num_pcd_per_seg, 3)
        sampled_segment_centers = sampled_segment_coords.mean(1)

        if self.cfg.segment_fea == 'pointnet':
            local_coords = sampled_segment_coords - sampled_segment_centers.unsqueeze(1)
            seg_points_feas = torch.cat([sampled_segment_coords, output_feats[seg_pcd_idxs]], -1).transpose(1,
                                                                                                            2).contiguous()
            seg_feas = self.segment_layer(xyz=local_coords, points=seg_points_feas)


        g = dgl.graph((spg_edgs[:, 0], spg_edgs[:, 1]), num_nodes=len(seg_feas))
        g = add_reverse_edges(g, copy_ndata=True, copy_edata=True)


        seg_sem_scores = self.segsem_layer(g, seg_feas)

        normal_labels = other_input['over_seg_label']
        semantic_preds = (seg_sem_scores[normal_labels]).max(1)[
            1]  # (N), long  Use spg sem prediction to replace point predictions

        ret['seg_sem_scores'] = seg_sem_scores
        ret['semantic_scores'] = semantic_scores

        if (epoch > self.prepare_epochs):
            #### get prooposal clusters
            if self.cfg.dataset == 'scannetv2':
                object_idxs = torch.nonzero(semantic_preds > 1).view(-1)  # except wall and floor
            elif self.cfg.dataset == '3rscan':
                if_fg = torch.logical_and(semantic_preds > 1,
                                          semantic_preds != 14)  # excep wall floor and ceiling(class index 14)
                object_idxs = torch.nonzero(if_fg).view(-1)
            else:
                raise KeyError
            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = utils.get_batch_offsets(batch_idxs_, input.batch_size)
            coords_ = coords[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]

            semantic_preds_cpu = semantic_preds[object_idxs].int().cpu()

            idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_,
                                                                          batch_offsets_, self.cluster_radius,
                                                                          self.cluster_shift_meanActive)
            proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(semantic_preds_cpu,
                                                                                     idx_shift.cpu(),
                                                                                     start_len_shift.cpu(),
                                                                                     self.cluster_npoint_thre)

            proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()


            if 'only_use_shited_coords' in self.cfg and self.cfg.only_use_shited_coords:
                proposals_idx = proposals_idx_shift
                proposals_offset = proposals_offset_shift
            else:
                idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_,
                                                                  self.cluster_radius,
                                                                  self.cluster_meanActive)
                proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx.cpu(),
                                                                             start_len.cpu(),
                                                                             self.cluster_npoint_thre)
                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
                # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int

                proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
                proposals_offset_shift += proposals_offset[-1]
                proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)
                proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]))

            #### proposals voxelization again
            input_feats, inp_map = self.clusters_voxelization(proposals_idx, proposals_offset, output_feats, coords,
                                                              self.score_fullscale, self.score_scale, self.mode)

            #### score
            score = self.score_unet(input_feats)
            score = self.score_outputlayer(score)
            score_feats = score.features[inp_map.long()]  # (sumNPoint, C)
            score_feats = pointgroup_ops.roipool(score_feats, proposals_offset.cuda())  # (nProposal, C)
            scores = self.score_linear(score_feats)  # (nProposal, 1)

            ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

        return ret




def model_fn_decorator(cfg, test=False):
    device = torch.device(f'cuda:{cfg.local_rank}')
    torch.cuda.set_device(device)
    #### criterion
    semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    score_criterion = nn.BCELoss(reduction='none').cuda()

    def test_model_fn(batch, model, epoch):
        coords = batch['locs'].cuda()  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()  # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()  # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda
        feats = batch['feats'].cuda()  # (N, C), float32, cuda

        batch_offsets = batch['offsets'].cuda()  # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        ### used for superpoint graph
        spatial_shape = batch['spatial_shape']
        spg_edgs = batch['spg_edgs'].cuda()
        spg_edg_offsets = batch['spg_edg_offsets'].cuda()
        seg_pcd_idxs = batch['seg_pcd_idxs'].cuda()
        uni_seg_labels = batch['uni_seg_labels']
        spg_edg_idxs = batch['spg_edg_idxs'].cuda()

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)
        if 'use_normal' in cfg:
            if cfg.use_normal:
                normals = batch['normals'].cuda()
                feats = torch.cat((feats, normals), 1)

        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        other_input = {}
        if cfg.use_overseg:
            other_input['over_seg_label'] = batch['normal_seg_labels']

        other_input['over_seg_label'] = batch['normal_seg_labels']
        other_input['seg_offsets'] = batch['seg_offsets']

        ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, spg_edgs, spg_edg_offsets,
                    seg_pcd_idxs, uni_seg_labels, spg_edg_idxs, epoch,
                    other_input=other_input)
        semantic_scores = ret['semantic_scores']  # (N, nClass) float32, cuda
        pt_offsets = ret['pt_offsets']  # (N, 3), float32, cuda
        if (epoch > cfg.prepare_epochs):
            scores, proposals_idx, proposals_offset = ret['proposal_scores']
        seg_sem_scores = ret['seg_sem_scores']

        ##### preds
        with torch.no_grad():
            preds = {}
            preds['semantic'] = seg_sem_scores[batch['normal_seg_labels']]
            # preds['semantic'] = semantic_scores

            preds['pt_offsets'] = pt_offsets
            if (epoch > cfg.prepare_epochs):
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)
            preds['spg_edgs'] = spg_edgs
        return preds

    def model_fn(batch, model, epoch):
        ##### prepare input and forward
        coords = batch['locs'].cuda()  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()  # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()  # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda
        feats = batch['feats'].cuda()  # (N, C), float32, cuda
        labels = batch['labels'].cuda()  # (N), long, cuda
        instance_labels = batch['instance_labels'].cuda()  # (N), long, cuda, 0~total_nInst, -100

        instance_info = batch['instance_info'].cuda()  # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda

        batch_offsets = batch['offsets'].cuda()  # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']
        spg_edgs = batch['spg_edgs'].cuda()
        spg_edg_offsets = batch['spg_edg_offsets'].cuda()
        seg_pcd_idxs = batch['seg_pcd_idxs'].cuda()
        uni_seg_labels = batch['uni_seg_labels']
        spg_edg_idxs = batch['spg_edg_idxs'].cuda()
        spg_node_sems = batch['spg_node_sems'].cuda()

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)

        if 'use_normal' in cfg:
            if cfg.use_normal:
                normals = batch['normals'].cuda()
                feats = torch.cat((feats, normals), 1)

        other_input = {}
        other_input['over_seg_label'] = batch['normal_seg_labels']
        other_input['seg_offsets'] = batch['seg_offsets']
        if cfg.use_overseg:
            other_input['labels'] = batch['labels']  # used to filter non-object over-segments

        if 'spg_edg_feas' in batch:
            other_input['spg_edg_feas'] = batch['spg_edg_feas'].cuda()

        if 'sp_geo_feas' in batch:
            other_input['sp_geo_feas'] = batch['sp_geo_feas'].cuda()

        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, spg_edgs, spg_edg_offsets,
                    seg_pcd_idxs, uni_seg_labels, spg_edg_idxs, epoch,
                    other_input=other_input)

        semantic_scores = ret['semantic_scores']  # (N, nClass) float32, cuda
        pt_offsets = ret['pt_offsets']  # (N, 3), float32, cuda
        seg_sem_scores = ret['seg_sem_scores']

        if (epoch > cfg.prepare_epochs):
            scores, proposals_idx, proposals_offset = ret['proposal_scores']
            # scores: (nProposal, 1) float, cuda
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu

        loss_inp = {}
        loss_inp['semantic_scores'] = (seg_sem_scores[batch['normal_seg_labels']], labels)
        loss_inp['pt_offsets'] = (pt_offsets, coords_float, instance_info, instance_labels)
        loss_inp['spg_edg_labels'] = batch['spg_edg_labels'].cuda()
        loss_inp['seg_sem_scores'] = (seg_sem_scores, spg_node_sems)
        loss_inp['normal_seg_labels'] = batch['normal_seg_labels']

        if (epoch > cfg.prepare_epochs):
            loss_inp['proposal_scores'] = (scores, proposals_idx, proposals_offset, instance_pointnum)
        loss, loss_out, infos = loss_fn(loss_inp, epoch)

        ##### accuracy / visual_dict / meter_dict
        with torch.no_grad():
            preds = {}
            preds['semantic'] = seg_sem_scores[batch['normal_seg_labels']]
            preds['pt_offsets'] = pt_offsets
            if (epoch > cfg.prepare_epochs):
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)
            preds['spg_edgs'] = spg_edgs
            visual_dict = {}
            visual_dict['loss'] = loss
            for k, v in loss_out.items():
                visual_dict[k] = v[0]

            meter_dict = {}
            meter_dict['loss'] = (loss.item(), coords.shape[0])

            for k, v in loss_out.items():
                meter_dict[k] = (float(v[0]), v[1])

            ### semantic predicton accuracy
            semantic_preds = preds['semantic'].max(1)[1]
            sem_valid_mask = labels >= 0
            correct = labels[sem_valid_mask] == semantic_preds[sem_valid_mask]
            sem_acc = correct.sum() / float(len(correct))
            meter_dict['sem_acc'] = (sem_acc, len(correct))
            print('Semantic prediction acc={:.4f}'.format(sem_acc.item()))
        return loss, preds, visual_dict, meter_dict

    def loss_fn(loss_inp, epoch):

        loss_out = {}
        infos = {}
        '''semantic loss'''
        semantic_scores, semantic_labels = loss_inp['semantic_scores']
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda
        semantic_loss = semantic_criterion(semantic_scores, semantic_labels)
        loss_out['semantic_loss'] = (semantic_loss, semantic_scores.shape[0])


        '''offset loss'''
        pt_offsets, coords, instance_info, instance_labels = loss_inp['pt_offsets']
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
        # instance_labels: (N), long

        gt_offsets = instance_info[:, 0:3] - coords  # (N, 3)
        pt_diff = pt_offsets - gt_offsets  # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # (N)
        valid = (instance_labels != cfg.ignore_label).float()
        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)  # (N)
        offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

        loss_out['offset_norm_loss'] = (offset_norm_loss, valid.sum())
        loss_out['offset_dir_loss'] = (offset_dir_loss, valid.sum())

        print("Semantic Loss", semantic_loss)

        if (epoch > cfg.prepare_epochs):
            '''score loss'''
            scores, proposals_idx, proposals_offset, instance_pointnum = loss_inp['proposal_scores']
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int

            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_labels,
                                          instance_pointnum)  # (nProposal, nInstance), float
            gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
            gt_scores = get_segmented_scores(gt_ious, cfg.fg_thresh, cfg.bg_thresh)

            score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            score_loss = score_loss.mean()

            loss_out['score_loss'] = (score_loss, gt_ious.shape[0])

        '''total loss'''
        loss = cfg.loss_weight[0] * semantic_loss + cfg.loss_weight[1] * offset_norm_loss + cfg.loss_weight[
            2] * offset_dir_loss

        if (epoch > cfg.prepare_epochs):
            loss += (cfg.loss_weight[3] * score_loss)

        return loss, loss_out, infos

    def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
        '''
        :param scores: (N), float, 0~1
        :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
        '''
        fg_mask = scores > fg_thresh
        bg_mask = scores < bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)

        segmented_scores = (fg_mask > 0).float()
        k = 1 / (fg_thresh - bg_thresh)
        b = bg_thresh / (bg_thresh - fg_thresh)
        segmented_scores[interval_mask] = scores[interval_mask] * k + b

        return segmented_scores

    if test:
        fn = test_model_fn
    else:
        fn = model_fn
    return fn


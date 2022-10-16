'''
3RScan Dataloader (Modified from SparseConvNet Dataloader)

'''
import os, sys, glob, math, numpy as np
import scipy.ndimage
import scipy.interpolate
import json
import torch
from torch.utils.data import DataLoader
import SharedArray as SA
import time
import random
import networkx as nx
import copy
import numpy as np
from collections import Counter
sys.path.append('../')
from lib.pointgroup_ops.functions import pointgroup_ops


class LabelCompacter:
    """
    Make labels in range[0, num_unique_labels]

    example:
        import numpy as np
        mapper = LabelCompacter()
        labels1 = np.random.randint(low=5876, high=9878, size=10, dtype=int)
        mapped_labels1 = mapper.map(labels1)
        labels2 = np.random.randint(low=9999, high=19999, size=10, dtype=int)
        mapped_labels2 = mapper.map(labels2)

    """

    def __init__(self):
        self.num_labels = 0
        self.label2localid = {}
        self.localid2label = {}

    def update_mapper_by_labels(self, labels: np.array):
        """
        Note that input labels of every calling are non-overlapping.
        :param labels:
        :return:
        """
        uni_labels = np.unique(labels)
        localids = np.arange(0, len(uni_labels)) + self.num_labels
        self.num_labels += len(uni_labels)
        label2localid_ = dict(zip(uni_labels.tolist(), localids.tolist()))
        localid2label_ = dict(zip(localids.tolist(), uni_labels.tolist()))
        self.label2localid.update(label2localid_)
        self.localid2label.update(localid2label_)

        # mapped_labels = np.zeros_like(labels)
        mapped_labels = np.vectorize(label2localid_.get)(labels)
        return mapped_labels

    def convert2localid(self, labels: np.array):
        mapped_labels = np.vectorize(self.label2localid.get)(labels)
        return mapped_labels

    def convertlocalid2segid(self, localids: np.array):
        segids = np.vectorize(self.localid2label.get)(localids)
        return segids



class Dataset:
    def __init__(self, cfg, test=False):
        self.data_root = cfg.data_root
        self.dataset = cfg.dataset
        self.filename_suffix = cfg.filename_suffix

        self.batch_size = cfg.batch_size
        self.train_workers = cfg.train_workers
        self.val_workers = cfg.train_workers

        self.full_scale = cfg.full_scale
        self.scale = cfg.scale
        self.max_npoint = cfg.max_npoint
        self.mode = cfg.mode

        self.dist = cfg.dist
        self.cache = cfg.cache
        self.cfg = cfg

        if test:
            self.test_split = cfg.split  # val or test
            self.test_workers = cfg.test_workers
            cfg.batch_size = 1


    def trainLoader(self):
        self.train_file_names = sorted(
            glob.glob(os.path.join(self.data_root, self.dataset, self.cfg.train_dir,
                                   '*' + self.filename_suffix)))  # train, train_backup, train_normal_seg

        self.train_file_names = [name for name in self.train_file_names if '4a9a43d8-7736-2874-86fc-098deb94c868' not in name]


        if not self.cache:
            self.train_files = [torch.load(i) for i in self.train_file_names]

        train_set = list(range(len(self.train_file_names)))
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if self.dist else None
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge,
                                            num_workers=self.train_workers,
                                            shuffle=(self.train_sampler is None), sampler=self.train_sampler,
                                            drop_last=True, pin_memory=True,
                                            worker_init_fn=self._worker_init_fn_)


    def valLoader(self):
        self.val_file_names = sorted(
            glob.glob(os.path.join(self.data_root, self.dataset, self.cfg.val_dir,
                                   '*' + self.filename_suffix)))  # val
        if not self.cache:
            self.val_files = [torch.load(i) for i in self.val_file_names]
        val_set = list(range(len(self.val_file_names)))
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(val_set) if self.dist else None

        self.val_data_loader = DataLoader(val_set, batch_size=1, collate_fn=self.valMerge,
                                          num_workers=self.val_workers,
                                          shuffle=False, sampler=self.val_sampler, drop_last=False, pin_memory=True,
                                          worker_init_fn=self._worker_init_fn_)
    def testLoader(self):
        self.test_file_names = sorted(
            glob.glob(os.path.join(self.data_root, self.dataset, self.cfg.test_dir,
                                   '*' + self.filename_suffix)))  # val

        if not self.cache:
            self.test_files = [torch.load(i) for i in self.test_file_names]

        test_set = list(np.arange(len(self.test_file_names)))


        self.test_data_loader = DataLoader(test_set, batch_size=1, collate_fn=self.testMerge,
                                           num_workers=self.cfg.test_workers,
                                           shuffle=False, drop_last=False, pin_memory=True)

    def _worker_init_fn_(self, worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        np.random.seed(np_seed)

    # Elastic distortion
    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

        def g(x_):
            return np.hstack([i(x_)[:, None] for i in interp])

        return x + g(x) * mag

    def getInstanceInfo(self, xyz, instance_label, sem_labels=None):
        '''
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        '''
        instance_info = np.ones((xyz.shape[0], 9),
                                dtype=np.float32) * -100.0  # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []  # (nInst), int
        instance_num = int(instance_label.max()) + 1
        inst_sem_labels = []
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)

            ### instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i

            ### instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)

            ### instance semantic label
            if sem_labels is not None:
                inst_sem_labels.append(sem_labels[inst_idx_i[0][0]])

        return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum, "inst_sem_labels": inst_sem_labels}

    def dataAugment(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0],
                              [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)

    def crop(self, xyz):
        '''
        :param xyz: (n, 3) >= 0
        '''
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.max_npoint):
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs

    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        # sometimes instance_label is empty
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def index_pcd_by_seglabels(self, seg_labels: np.array, sample_seg_size: int, start_idx: int = 0):
        """

        :param seg_labels: shape ()
        :param sample_seg_size: num of points to sample per segment
        :param start_idx:  the idx of this sample's first point in the whole batch
        :return:
            seg_pcd_idxs: shape (num_segs, sample_seg_size), each element is an index of pointcloud
            uni_seg_labels: shape (num_segs)
        """
        seg_pcd_idxs = []
        uni_seg_labels = np.array(sorted(np.unique(seg_labels)))
        for seg in uni_seg_labels:
            seg_idxs = np.where(seg_labels == seg)[0]
            seg_idxs = seg_idxs + start_idx
            np.random.shuffle(seg_idxs)
            seg_pcd_idxs.append(np.random.choice(seg_idxs, sample_seg_size))
        seg_pcd_idxs = np.array(seg_pcd_idxs)

        return seg_pcd_idxs, uni_seg_labels

    def gen_edge_index(self, spg_edgs: np.array, uni_seg_labels: np.array):
        """
            Get edge's node(segment) index among segments in a batch.
        :param spg_edgs:
        :param uni_seg_labels:
        :return:
            edg_idxs: np.array, shape (num_edgs, 2)
        """
        seglabel2idx = {seg: i for i, seg in enumerate(uni_seg_labels)}
        edg_idxs = []
        for src, tgt in spg_edgs:
            edg_idxs.append([seglabel2idx[src], seglabel2idx[tgt]])
        return np.array(edg_idxs)

    def trainMerge(self, id):
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []
        normal_seg_labels = []  # generate implicit parts according to normal consistency

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int

        batch_offsets = [0]

        total_inst_num = 0
        seg2inst = {}
        total_time = 0
        spg_edgs = []  # shape will be num_total_segs * 2
        spg_seg_offsets = [0]
        spg_edg_offsets = [0]
        spg_edg_labels = []
        spg_edg_feas = []
        sp_geo_feas = []
        seg_pcd_idxs = []
        uni_seg_labels = []
        seg_offsets = [0]
        label_mapper = LabelCompacter()  # todo
        scene_names = []



        for i, idx in enumerate(id):
            # todo  optimize
            if self.cache:
                fn = self.train_file_names[idx].split('/')[-1].replace('.pth', '')
                xyz_origin = SA.attach("shm://{}_xyz".format(fn)).copy()
                label = SA.attach("shm://{}_label".format(fn)).copy()
                instance_label = SA.attach("shm://{}_instance_label".format(fn)).copy()
                normal_seg_label = SA.attach("shm://{}_normal_seg_label".format(fn)).copy()
                scene_names.append(fn)
            else:
                xyz_origin, label, instance_label, normal_seg_label = self.train_files[
                    idx]
                scene_names.append(self.train_files[idx])
            fn = self.train_file_names[idx].split('/')[-1].replace('.pth', '')

            xyz_origin = xyz_origin.astype(np.float32)
            label = label.astype(np.int32)
            instance_label = instance_label.astype(np.int32)
            normal_seg_label = normal_seg_label.astype(np.int32)
            rgb = np.zeros_like(xyz_origin)
            ### jitter / flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, True, True, True)
            ### scale
            xyz = xyz_middle * self.scale

            ### elastic
            xyz = self.elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
            xyz = self.elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)

            ### offset
            xyz -= xyz.min(0)

            ### crop
            # re-crop if there is no object after cropping
            xyz_backup = copy.copy(xyz)
            xyz, valid_idxs = self.crop(xyz_backup)
            while (valid_idxs.sum() == 0):
                print(fn, 'is empty after cropping, so re-cropping....')
                xyz, valid_idxs = self.crop(xyz_backup)
            while (instance_label[valid_idxs].max() < 0):
                print(fn, 'has no instance after cropping, so re-cropping....')
                xyz, valid_idxs = self.crop(xyz_backup)

            ### make sure num of segments in spg is samller than threshold

            if 'spg_postfix' in self.cfg:
                if self.cfg.max_num_segs > 0:
                    normal_seg_label = normal_seg_label.astype(np.int)
                    nodes_list = np.unique(normal_seg_label).astype(np.int).tolist()


                    if len(nodes_list) > self.cfg.max_num_segs:
                        spg_cut_valid_idxs = np.zeros_like(valid_idxs)
                        random_select = False
                        select_largest = True
                        if random_select:
                            random.shuffle(nodes_list)
                            remained_nodes = nodes_list[:min(len(nodes_list), self.cfg.max_num_segs)]
                        elif select_largest:
                            seg_cnt = Counter()
                            seg_cnt.update(normal_seg_label)
                            remained_nodes = [segid for segid,  segsize in seg_cnt.most_common(self.cfg.max_num_segs)]
                        else:
                            spg_path = os.path.join(self.cfg.data_root, self.cfg.dataset, self.cfg.train_dir,
                                                    fn + self.cfg.spg_postfix)
                            with open(spg_path) as f:
                                spg_graph = json.load(f)
                            G = nx.Graph()
                            G.add_nodes_from(nodes_list)
                            G.add_edges_from([(src, tgt, {'se_same_inst_label': se_label}) for src, tgt, se_label in
                                              zip(spg_graph['source'], spg_graph['target'], spg_graph['se_same_inst_label'])])


                            bfs_seed = random.choice(nodes_list)
                            bfs_tree = nx.bfs_tree(G, bfs_seed)
                            if len(bfs_tree) < 100:
                                bfs_seed = np.bincount(spg_graph['source'] + spg_graph['target']).argmax()
                                bfs_tree = nx.bfs_tree(G, bfs_seed)

                            remained_nodes = list(bfs_tree.nodes)[:min(len(G.nodes), self.cfg.max_num_segs)]

                        for segid in remained_nodes:
                            spg_cut_valid_idxs[normal_seg_label==segid] = 1
                        spg_cut_valid_idxs = spg_cut_valid_idxs.astype(np.bool)

                        print(fn, 'num_segs={}'.format(len(np.unique(normal_seg_label))),
                              'remain segs={}'.format(len(remained_nodes)), 'original_pcd={}, after_pcd={}'.format(len(spg_cut_valid_idxs), spg_cut_valid_idxs.sum()))

                        valid_idxs = np.logical_and(spg_cut_valid_idxs, valid_idxs)




            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

            normal_seg_label = normal_seg_label[valid_idxs]
            normal_seg_label = normal_seg_label + i * 100000000  # make sure that segid is different between scenes
            normal_seg_label = label_mapper.update_mapper_by_labels(normal_seg_label)  # todo
            for seg, inst_l in zip(normal_seg_label, instance_label):
                seg2inst[seg] = inst_l

            normal_seg_labels.append(torch.from_numpy(normal_seg_label))


            ### read superpoint graph from json
            if 'spg_postfix' in self.cfg:
                spg_path = os.path.join(self.cfg.data_root, self.cfg.dataset, self.cfg.train_dir,
                                        fn + self.cfg.spg_postfix)
                with open(spg_path) as f:
                    spg_graph = json.load(f)
                seg_id_set = set(normal_seg_label.tolist())

                source_ids = np.array(spg_graph['source']).astype(np.int) + i * 100000000  # todo
                target_ids = np.array(spg_graph['target']).astype(np.int) + i * 100000000  # todo
                edge_valid = [src_id in label_mapper.label2localid and tgt_id in label_mapper.label2localid for
                              src_id, tgt_id in zip(source_ids, target_ids)]
                edge_valid = np.array(edge_valid)
                source_ids = source_ids[edge_valid]
                target_ids = target_ids[edge_valid]
                edge_label = np.array(spg_graph['se_same_inst_label'])[edge_valid]



                source_ids = label_mapper.convert2localid(source_ids)  # todo
                target_ids = label_mapper.convert2localid(target_ids)  # todo
                src_nodes = torch.tensor(source_ids).long()  # todo
                tgt_nodes = torch.tensor(target_ids).long()  # todo
                se_same_inst_label = torch.tensor(edge_label).long()
                edges = torch.stack([src_nodes, tgt_nodes]).transpose(1, 0)  # todo

                spg_edgs.append(edges)
                spg_edg_labels.append(se_same_inst_label)
                spg_edg_offsets.append(spg_edg_offsets[-1] + len(edges))
                spg_seg_offsets.append(spg_seg_offsets[-1] + len(seg_id_set))


                seg_pcd_idx, uni_seg_label = self.index_pcd_by_seglabels(normal_seg_label,
                                                                         sample_seg_size=self.cfg.sample_seg_size,
                                                                         start_idx=batch_offsets[-1])
                seg_pcd_idxs.append(torch.from_numpy(seg_pcd_idx).long())
                uni_seg_labels.append(torch.from_numpy(uni_seg_label).long())
                seg_offsets.append(seg_offsets[-1]+len(uni_seg_label))


            ### get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32))
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]  # (nInst), list

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])
            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb))
            labels.append(torch.from_numpy(label))
            instance_labels.append(torch.from_numpy(instance_label))


            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)

        ### merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)
        seg_offsets = torch.tensor(seg_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)
        labels = torch.cat(labels, 0).long()  # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()  # long (N)
        if 'spg_postfix' in self.cfg:
            spg_edg_labels = torch.cat(spg_edg_labels, 0).long()  # long (num_valid_superedges)
        normal_seg_labels = torch.cat(normal_seg_labels, 0).long()  # long (N)
        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)  # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        return_dict = {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                       'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
                       'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
                       'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape, 'seg2inst': seg2inst}
        return_dict['scene_names'] = scene_names
        return_dict['normal_seg_labels'] = normal_seg_labels
        if 'spg_postfix' in self.cfg:
            spg_edgs = torch.cat(spg_edgs,
                                 0).long()  # shape: (num_batch_segs, 2) each line is an edge containing src_seg_id and tgt_seg_id
            spg_edg_offsets = torch.tensor(spg_edg_offsets).long()  # shape: (num_super_edges)
            spg_seg_offsets = torch.tensor(spg_seg_offsets).long()
            seg_pcd_idxs = torch.cat(seg_pcd_idxs, 0).long()
            uni_seg_labels = torch.cat(uni_seg_labels, 0).long()
            if len(spg_edg_feas) > 0:
                spg_edg_feas = torch.cat(spg_edg_feas, 0)
                return_dict['spg_edg_feas'] = spg_edg_feas

            if len(sp_geo_feas) > 0:
                sp_geo_feas = torch.cat(sp_geo_feas, 0)
                return_dict['sp_geo_feas'] = sp_geo_feas

            return_dict['spg_edgs'] = spg_edgs
            return_dict['spg_edg_offsets'] = spg_edg_offsets
            return_dict['spg_seg_offsets'] = spg_seg_offsets
            return_dict['spg_edg_labels'] = spg_edg_labels
            return_dict['seg_pcd_idxs'] = seg_pcd_idxs
            return_dict['uni_seg_labels'] = uni_seg_labels
            return_dict['uni_seg_labels_global'] = torch.from_numpy(
                label_mapper.convertlocalid2segid(uni_seg_labels.numpy())).long()

            spg_edg_idxs = self.gen_edge_index(spg_edgs.numpy(), uni_seg_labels.numpy())
            spg_edg_idxs = torch.from_numpy(spg_edg_idxs).long()
            return_dict['spg_edg_idxs'] = spg_edg_idxs

            ### use original implicit part index
            return_dict['spg_edgs_global'] = torch.from_numpy(
                label_mapper.convertlocalid2segid(spg_edgs.numpy())).long()
            return_dict['normal_seg_labels_global'] = torch.from_numpy(
                label_mapper.convertlocalid2segid(normal_seg_labels.numpy())).long()


            ### implicit part's semantic label
            segid2sem = dict(zip(normal_seg_labels.numpy().tolist(), labels.numpy().tolist()))
            spg_node_sems = np.vectorize(segid2sem.get)(uni_seg_labels.numpy())
            return_dict['spg_node_sems'] = torch.from_numpy(spg_node_sems).long()

            ### implicit part's instance label
            segid2inst = dict(zip(normal_seg_labels.numpy().tolist(), instance_labels.numpy().tolist()))
            spg_node_inst = np.vectorize(segid2inst.get)(uni_seg_labels.numpy())
            return_dict['seg_inst_label'] = torch.from_numpy(spg_node_inst).long()
            return_dict['seg_offsets'] = seg_offsets

        return return_dict

    def valMerge(self, id):
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []
        normal_seg_labels = []  # segment input mesh into implicit parts according to normal consistency

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int

        batch_offsets = [0]

        total_inst_num = 0
        seg2inst = {}
        total_time = 0
        spg_edgs = []  # shape will be num_total_segs * 2
        spg_seg_offsets = [0]
        spg_edg_offsets = [0]
        spg_edg_labels = []
        spg_edg_feas = []
        seg_pcd_idxs = []
        uni_seg_labels = []
        seg_offsets = [0]
        scene_names = []
        sp_geo_feas = []

        label_mapper = LabelCompacter()  # todo
        for i, idx in enumerate(id):
            if self.cache:
                fn = self.val_file_names[idx].split('/')[-1].replace('.pth', '')
                xyz_origin = SA.attach("shm://{}_xyz".format(fn)).copy()
                label = SA.attach("shm://{}_label".format(fn)).copy()
                instance_label = SA.attach("shm://{}_instance_label".format(fn)).copy()
                normal_seg_label = SA.attach("shm://{}_normal_seg_label".format(fn)).copy()
            else:
                xyz_origin, label, instance_label, normal_seg_label = self.val_files[idx]
            xyz_origin = xyz_origin.astype(np.float32)
            label = label.astype(np.int32)
            instance_label = instance_label.astype(np.int32)
            normal_seg_label = normal_seg_label.astype(np.int32)
            rgb = np.zeros_like(xyz_origin)
            fn = self.val_file_names[idx].split('/')[-1].replace('.pth', '')
            scene_names.append(fn)

            for seg, inst_l in zip(normal_seg_labels, instance_label):
                seg2inst[seg] = inst_l
            ### flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, False, True, True)

            ### scale
            xyz = xyz_middle * self.scale

            ### offset
            xyz -= xyz.min(0)

            ### crop
            # re-crop if there is no object after cropping
            xyz_backup = copy.copy(xyz)
            xyz, valid_idxs = self.crop(xyz_backup)
            while (valid_idxs.sum() == 0):
                print(fn, 'is empty after cropping, so re-cropping....')
                xyz, valid_idxs = self.crop(xyz_backup)
            while (instance_label[valid_idxs].max() < 0):
                print(fn, 'has no instance after cropping, so re-cropping....')
                xyz, valid_idxs = self.crop(xyz_backup)


            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

            normal_seg_label = normal_seg_label[valid_idxs]
            normal_seg_label = normal_seg_label + i * 100000000
            normal_seg_label = label_mapper.update_mapper_by_labels(normal_seg_label)
            for seg, inst_l in zip(normal_seg_label, instance_label):
                seg2inst[seg] = inst_l

            normal_seg_labels.append(torch.from_numpy(normal_seg_label))


            ### read superpoint graph from json
            if 'spg_postfix' in self.cfg:
                spg_path = os.path.join(self.cfg.data_root, self.cfg.dataset, self.cfg.val_dir,
                                        fn + self.cfg.spg_postfix)
                with open(spg_path) as f:
                    spg_graph = json.load(f)
                # seg_id_set = set(torch.unique(normal_seg_label).numpy().tolist())
                seg_id_set = set(normal_seg_label.tolist())

                source_ids = np.array(spg_graph['source']).astype(np.int) + i * 100000000
                target_ids = np.array(spg_graph['target']).astype(np.int) + i * 100000000
                edge_valid = [src_id in label_mapper.label2localid and tgt_id in label_mapper.label2localid for
                              src_id, tgt_id in zip(source_ids, target_ids)]
                edge_valid = np.array(edge_valid)
                source_ids = source_ids[edge_valid]
                target_ids = target_ids[edge_valid]
                edge_label = np.array(spg_graph['se_same_inst_label'])[edge_valid]
                source_ids = label_mapper.convert2localid(source_ids)
                target_ids = label_mapper.convert2localid(target_ids)
                src_nodes = torch.tensor(source_ids).long()
                tgt_nodes = torch.tensor(target_ids).long()
                se_same_inst_label = torch.tensor(edge_label).long()
                edges = torch.stack([src_nodes, tgt_nodes]).transpose(1, 0)

                spg_edgs.append(edges)
                spg_edg_labels.append(se_same_inst_label)
                spg_edg_offsets.append(spg_edg_offsets[-1] + len(edges))

                spg_seg_offsets.append(spg_seg_offsets[-1] + len(seg_id_set))

                seg_pcd_idx, uni_seg_label = self.index_pcd_by_seglabels(normal_seg_label, sample_seg_size=128,
                                                                         start_idx=batch_offsets[-1])
                seg_pcd_idxs.append(torch.from_numpy(seg_pcd_idx).long())
                uni_seg_labels.append(torch.from_numpy(uni_seg_label).long())
                seg_offsets.append(seg_offsets[-1]+len(uni_seg_label))


            ### get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32))
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]  # (nInst), list

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb))
            labels.append(torch.from_numpy(label))
            instance_labels.append(torch.from_numpy(instance_label))

            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)

        ### merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)
        seg_offsets = torch.tensor(seg_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)
        labels = torch.cat(labels, 0).long()  # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()  # long (N)


        normal_seg_labels = torch.cat(normal_seg_labels, 0).long()  # long (N)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)  # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)
        return_dict = {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                       'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
                       'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
                       'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape, 'seg2inst': seg2inst}
        return_dict['normal_seg_labels'] = normal_seg_labels

        if 'spg_postfix' in self.cfg:
            spg_edg_labels = torch.cat(spg_edg_labels, 0).long()  # long (num_valid_superedges)

            spg_edgs = torch.cat(spg_edgs,
                                 0).long()  # shape: (num_batch_segs, 2) each line is an edge containing src_seg_id and tgt_seg_id
            spg_edg_offsets = torch.tensor(spg_edg_offsets).long()  # shape: (num_super_edges)
            spg_seg_offsets = torch.tensor(spg_seg_offsets).long()
            seg_pcd_idxs = torch.cat(seg_pcd_idxs, 0).long()
            uni_seg_labels = torch.cat(uni_seg_labels, 0).long()

            if len(spg_edg_feas) > 0:
                spg_edg_feas = torch.cat(spg_edg_feas, 0)
                return_dict['spg_edg_feas'] = spg_edg_feas

            if len(sp_geo_feas) > 0:
                sp_geo_feas = torch.cat(sp_geo_feas, 0)
                return_dict['sp_geo_feas'] = sp_geo_feas

            return_dict['spg_edgs'] = spg_edgs
            return_dict['spg_edg_offsets'] = spg_edg_offsets
            return_dict['spg_seg_offsets'] = spg_seg_offsets
            return_dict['spg_edg_labels'] = spg_edg_labels
            return_dict['seg_pcd_idxs'] = seg_pcd_idxs
            return_dict['uni_seg_labels'] = uni_seg_labels

            spg_edg_idxs = self.gen_edge_index(spg_edgs.numpy(), uni_seg_labels.numpy())
            spg_edg_idxs = torch.from_numpy(spg_edg_idxs).long()
            return_dict['spg_edg_idxs'] = spg_edg_idxs

            ### use original segment index
            return_dict['spg_edgs_global'] = torch.from_numpy(
                label_mapper.convertlocalid2segid(spg_edgs.numpy())).long()
            return_dict['normal_seg_labels_global'] = torch.from_numpy(
                label_mapper.convertlocalid2segid(normal_seg_labels.numpy())).long()
            return_dict['uni_seg_labels_global'] = torch.from_numpy(
                label_mapper.convertlocalid2segid(uni_seg_labels.numpy())).long()
            ### implicit part's semantic label
            segid2sem = dict(zip(normal_seg_labels.numpy().tolist(), labels.numpy().tolist()))
            spg_node_sems = np.vectorize(segid2sem.get)(uni_seg_labels.numpy())
            return_dict['spg_node_sems'] = torch.from_numpy(spg_node_sems).long()

            ### implicit part's instance label
            segid2inst = dict(zip(normal_seg_labels.numpy().tolist(), instance_labels.numpy().tolist()))
            spg_node_inst = np.vectorize(segid2inst.get)(uni_seg_labels.numpy())
            return_dict['seg_inst_label'] = torch.from_numpy(spg_node_inst).long()
            return_dict['seg_offsets'] = seg_offsets

        return return_dict

    def testMerge(self, id):
        locs = []
        locs_float = []
        feats = []
        labels = []  # used for calculation semantic classification acc.
        normal_seg_labels = []  # segment input mesh into implicit parts according to normal consistency
        instance_labels = []

        batch_offsets = [0]
        spg_edgs = []  # shape will be num_total_segs * 2
        spg_seg_offsets = [0]
        spg_edg_offsets = [0]
        seg_pcd_idxs = []
        uni_seg_labels = []
        spg_edg_labels = []
        label_mapper = LabelCompacter()  # todo
        seg_offsets = [0]
        scene_names = []
        spg_edg_feas = []
        sp_geo_feas = []

        for i, idx in enumerate(id):
            assert self.test_split in ['val', 'test']
            if not self.cache:
                xyz_origin, label, instance_label, normal_seg_label = self.test_files[idx]
            else:
                fn = self.test_file_names[idx].split('/')[-1].replace('.pth', '')
                xyz_origin = SA.attach("shm://{}_xyz".format(fn)).copy()

                normal_seg_label = SA.attach("shm://{}_normal_seg_label".format(fn)).copy()
                label = SA.attach("shm://{}_label".format(fn)).copy()
                instance_label = SA.attach("shm://{}_instance_label".format(fn)).copy()
            xyz_origin = xyz_origin.astype(np.float32)
            label = label.astype(np.int32)
            instance_label = instance_label.astype(np.int32)
            normal_seg_label = normal_seg_label.astype(np.int32)
            rgb = np.zeros_like(xyz_origin)


            fn = self.test_file_names[idx].split('/')[-1].replace('.pth', '')
            scene_names.append(fn)
            normal_seg_label = normal_seg_label + i * 100000000
            normal_seg_label = label_mapper.update_mapper_by_labels(normal_seg_label)  # todo

            ### flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, False, True, True)

            ### scale
            xyz = xyz_middle * self.scale

            ### offset
            xyz -= xyz.min(0)

            ### read superpoint graph from json
            if 'spg_postfix' in self.cfg:
                spg_path = os.path.join(self.cfg.data_root, self.cfg.dataset, self.cfg.test_dir,
                                        fn + self.cfg.spg_postfix)
                with open(spg_path) as f:
                    spg_graph = json.load(f)
                # seg_id_set = set(torch.unique(normal_seg_label).numpy().tolist())
                seg_id_set = set(normal_seg_label.tolist())



                source_ids = np.array(spg_graph['source']).astype(np.int) + i * 100000000  # todo
                target_ids = np.array(spg_graph['target']).astype(np.int) + i * 100000000  # todo
                edge_valid = [src_id in label_mapper.label2localid and tgt_id in label_mapper.label2localid for
                              src_id, tgt_id in zip(source_ids, target_ids)]
                edge_valid = np.array(edge_valid)
                source_ids = source_ids[edge_valid]
                target_ids = target_ids[edge_valid]
                edge_label = np.array(spg_graph['se_same_inst_label'])[edge_valid]


                source_ids = label_mapper.convert2localid(source_ids)  # todo
                target_ids = label_mapper.convert2localid(target_ids)  # todo
                src_nodes = torch.tensor(source_ids).long()  # todo
                tgt_nodes = torch.tensor(target_ids).long()  # todo
                edges = torch.stack([src_nodes, tgt_nodes]).transpose(1, 0)

                se_same_inst_label = torch.tensor(edge_label).long()

                valid_edges = []
                for src, tgt in edges.numpy().tolist():
                    if src in seg_id_set and tgt in seg_id_set:
                        valid_edges.append([src, tgt])
                valid_edges = torch.tensor(valid_edges).long()

                spg_edgs.append(valid_edges)
                spg_edg_offsets.append(spg_edg_offsets[-1] + len(valid_edges))
                spg_seg_offsets.append(spg_seg_offsets[-1] + len(seg_id_set))


                seg_pcd_idx, uni_seg_label = self.index_pcd_by_seglabels(normal_seg_label, sample_seg_size=128,
                                                                         start_idx=batch_offsets[-1])
                seg_pcd_idxs.append(torch.from_numpy(seg_pcd_idx).long())
                uni_seg_labels.append(torch.from_numpy(uni_seg_label).long())

                spg_edg_labels.append(se_same_inst_label)
                seg_offsets.append(seg_offsets[-1]+len(uni_seg_label))


            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb))

            normal_seg_labels.append(torch.from_numpy(normal_seg_label))

            labels.append(torch.from_numpy(label))
            instance_labels.append(torch.from_numpy(instance_label))

        ### merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)
        seg_offsets = torch.tensor(seg_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)
        instance_labels = torch.cat(instance_labels, 0).long()  # long (N)
        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)
        normal_seg_labels = torch.cat(normal_seg_labels, 0).long()  # long (N)


        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)
        labels = torch.cat(labels, 0).long()


        return_dict = {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                       'locs_float': locs_float, 'feats': feats,
                       'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}
        return_dict['normal_seg_labels'] = normal_seg_labels
        return_dict['labels'] = labels
        return_dict['instance_labels'] = instance_labels
        return_dict['scene_names'] = scene_names


        if 'spg_postfix' in self.cfg:
            spg_edg_labels = torch.cat(spg_edg_labels, 0).long()  # long (num_valid_superedges)
            spg_edgs = torch.cat(spg_edgs,
                                 0).long()  # shape: (num_batch_segs, 2) each line is an edge containing src_seg_id and tgt_seg_id
            spg_edg_offsets = torch.tensor(spg_edg_offsets).long()  # shape: (num_super_edges)
            spg_seg_offsets = torch.tensor(spg_seg_offsets).long()
            seg_pcd_idxs = torch.cat(seg_pcd_idxs, 0).long()
            uni_seg_labels = torch.cat(uni_seg_labels, 0).long()

            if len(spg_edg_feas) > 0:
                spg_edg_feas = torch.cat(spg_edg_feas, 0)
                return_dict['spg_edg_feas'] = spg_edg_feas

            if len(sp_geo_feas) > 0:
                sp_geo_feas = torch.cat(sp_geo_feas, 0)
                return_dict['sp_geo_feas'] = sp_geo_feas

            return_dict['spg_edgs'] = spg_edgs
            return_dict['spg_edg_offsets'] = spg_edg_offsets
            return_dict['spg_seg_offsets'] = spg_seg_offsets
            return_dict['seg_pcd_idxs'] = seg_pcd_idxs
            return_dict['uni_seg_labels'] = uni_seg_labels

            return_dict['spg_edg_labels'] = spg_edg_labels

            spg_edg_idxs = self.gen_edge_index(spg_edgs.numpy(), uni_seg_labels.numpy())
            spg_edg_idxs = torch.from_numpy(spg_edg_idxs).long()
            return_dict['spg_edg_idxs'] = spg_edg_idxs

            ### use original segment index
            return_dict['spg_edgs_global'] = torch.from_numpy(
                label_mapper.convertlocalid2segid(spg_edgs.numpy())).long()
            return_dict['normal_seg_labels_global'] = torch.from_numpy(
                label_mapper.convertlocalid2segid(normal_seg_labels.numpy())).long()
            return_dict['uni_seg_labels_global'] = torch.from_numpy(
                label_mapper.convertlocalid2segid(uni_seg_labels.numpy())).long()

            ### segment's semantic label
            segid2sem = dict(zip(normal_seg_labels.numpy().tolist(), labels.numpy().tolist()))
            spg_node_sems = np.vectorize(segid2sem.get)(uni_seg_labels.numpy())
            return_dict['spg_node_sems'] = torch.from_numpy(spg_node_sems).long()
            return_dict['seg_offsets'] = seg_offsets

        return return_dict



if __name__ == '__main__':
    # python threerscan_inst.py --config ../3rscan_config/pointgroupleaky_run2_3rscan.yaml
    import util.utils as utils

    def init():
        # config
        global cfg
        from util.config import get_parser
        cfg = get_parser()
        cfg.dist = False
        cfg.cache = False


        # random seed
        random.seed(cfg.manual_seed)
        np.random.seed(cfg.manual_seed)
        torch.manual_seed(cfg.manual_seed)
        torch.cuda.manual_seed_all(cfg.manual_seed)

    init()
    cfg.data_root = 'workspace/PointGroup/dataset/'
    dataset = Dataset(cfg, test=False)
    dataset.trainLoader()
    data = dataset.trainMerge([1])

    ##### visualization
    import seaborn as sns
    import matplotlib.pyplot as plt
    COLOR_PALETTE = sns.color_palette() + sns.color_palette("husl", 9) + sns.color_palette("pastel") + sns.color_palette("Set2")
    COLOR_PALETTE =  COLOR_PALETTE * 100
    rgb = np.array([COLOR_PALETTE[label] for label in data['instance_labels']])
    rgb[data['instance_labels'].numpy()==-100,:] = 0
    def save_pcd(pts, rgb):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.io.write_point_cloud('temp.ply', pcd)

    save_pcd(data['locs_float'], rgb)


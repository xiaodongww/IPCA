# ------------------------------------------------------------------------------
# ---------  Graph methods for SuperPoint Graph   ------------------------------
# ---------     Loic Landrieu, Dec. 2017     -----------------------------------
# ------------------------------------------------------------------------------
import numpy as np
from numpy.matlib import repmat
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from numpy import linalg as LA
from collections import defaultdict

import torch
import json
import glob
import os
import multiprocessing
import networkx as nx


# ------------------------------------------------------------------------------
def compute_sp_graph(xyz, d_max, in_component, components, seg_ids, labels, n_labels):
    """
    compute the superpoint graph with superpoints and superedges features
    :param xyz:
    :param d_max: max length of super edges
    :param in_component:  index of the component in which the node belong
    :param components:  n_nodes_red x 1 cell components : for each component, list of the nodes.  n_nodes_red: number of components
    :param seg_ids:  segment ids relevant to components.
    :param labels:  semantic lables.
    :param n_labels: number of classes
    :return:
    """
    # n_com = max(in_component) + 1
    n_com = len(np.unique(in_component))
    in_component = np.array(in_component)
    has_labels = len(labels) > 0
    label_hist = has_labels and len(labels.shape) > 1 and labels.shape[1] > 1
    # ---compute delaunay triangulation---
    tri = Delaunay(xyz)
    # interface select the edges between different components
    # edgx and edgxr converts from tetrahedrons to edges
    # done separatly for each edge of the tetrahedrons to limit memory impact
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 1]]
    edg1 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 1]))
    edg1r = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 2]]
    edg2 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 2]))
    edg2r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 3]]
    edg3 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 3]))
    edg3r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 2]]
    edg4 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 2]))
    edg4r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 1]))
    interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 3]]
    edg5 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 3]))
    edg5r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 1]))
    interface = in_component[tri.vertices[:, 2]] != in_component[tri.vertices[:, 3]]
    edg6 = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 3]))
    edg6r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 2]))
    del tri, interface
    edges = np.hstack((edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r,
                       edg3r, edg4r, edg5r, edg6r))
    del edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r, edg3r, edg4r, edg5r, edg6r
    edges = np.unique(edges, axis=1)

    if d_max > 0:
        dist = np.sqrt(((xyz[edges[0, :]] - xyz[edges[1, :]]) ** 2).sum(1))
        edges = edges[:, dist < d_max]
    # ---sort edges by alpha numeric order wrt to the components of their source/target---
    n_edg = len(edges[0])
    edge_comp = in_component[edges]
    edge_comp_index = n_com * edge_comp[0, :] + edge_comp[1, :]
    order = np.argsort(edge_comp_index)
    edges = edges[:, order]
    edge_comp = edge_comp[:, order]
    edge_comp_index = edge_comp_index[order]
    # marks where the edges change components iot compting them by blocks
    jump_edg = np.vstack((0, np.argwhere(np.diff(edge_comp_index)) + 1, n_edg)).flatten()
    n_sedg = len(jump_edg) - 1
    # ---set up the edges descriptors---
    graph = dict([("is_nn", False)])
    graph["segid2localid"] = {segid: localid for localid, segid in enumerate(seg_ids)}
    graph["localid2segid"] = {localid: segid for localid, segid in enumerate(seg_ids)}
    graph["sp_centroids"] = np.zeros((n_com, 3), dtype='float32')
    graph["sp_length"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_surface"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_volume"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_count"] = np.zeros((n_com, 1), dtype='uint64')
    graph["source"] = np.zeros((n_sedg, 1), dtype='uint32')
    graph["target"] = np.zeros((n_sedg, 1), dtype='uint32')
    graph["se_delta_mean"] = np.zeros((n_sedg, 3), dtype='float32')
    graph["se_delta_std"] = np.zeros((n_sedg, 3), dtype='float32')
    graph["se_delta_norm"] = np.zeros((n_sedg, 1), dtype='float32')
    graph["se_delta_centroid"] = np.zeros((n_sedg, 3), dtype='float32')
    graph["se_length_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    graph["se_surface_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    graph["se_volume_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    graph["se_point_count_ratio"] = np.zeros((n_sedg, 1), dtype='float32')

    if has_labels:
        graph["sp_labels"] = np.zeros((n_com, n_labels + 1), dtype='uint32')
    else:
        graph["sp_labels"] = []
    # ---compute the superpoint features---
    for i_com in range(0, n_com):
        comp = components[i_com]
        if has_labels and not label_hist:
            graph["sp_labels"][i_com, :] = np.histogram(labels[comp]
                                                        , bins=[float(i) - 0.5 for i in range(0, n_labels + 2)])[0]
        if has_labels and label_hist:
            graph["sp_labels"][i_com, :] = sum(labels[comp, :])
        graph["sp_point_count"][i_com] = len(comp)
        xyz_sp = np.unique(xyz[comp, :], axis=0)
        if len(xyz_sp) == 1:
            graph["sp_centroids"][i_com] = xyz_sp
            graph["sp_length"][i_com] = 0
            graph["sp_surface"][i_com] = 0
            graph["sp_volume"][i_com] = 0
        elif len(xyz_sp) == 2:
            graph["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)
            graph["sp_length"][i_com] = np.sqrt(np.sum(np.var(xyz_sp, axis=0)))
            graph["sp_surface"][i_com] = 0
            graph["sp_volume"][i_com] = 0
        else:
            ev = LA.eig(np.cov(np.transpose(xyz_sp), rowvar=True))
            ev = -np.sort(-ev[0])  # descending order
            graph["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)
            try:
                graph["sp_length"][i_com] = ev[0]
            except TypeError:
                graph["sp_length"][i_com] = 0
            try:
                graph["sp_surface"][i_com] = np.sqrt(ev[0] * ev[1] + 1e-10)
            except TypeError:
                graph["sp_surface"][i_com] = 0
            try:
                graph["sp_volume"][i_com] = np.sqrt(ev[0] * ev[1] * ev[2] + 1e-10)
            except TypeError:
                graph["sp_volume"][i_com] = 0
    # ---compute the superedges features---
    for i_sedg in range(0, n_sedg):
        i_edg_begin = jump_edg[i_sedg]
        i_edg_end = jump_edg[i_sedg + 1]
        ver_source = edges[0, range(i_edg_begin, i_edg_end)]
        ver_target = edges[1, range(i_edg_begin, i_edg_end)]
        com_source = edge_comp[0, i_edg_begin]
        com_target = edge_comp[1, i_edg_begin]

        com_source = graph["segid2localid"][com_source]
        com_target = graph["segid2localid"][com_target]

        xyz_source = xyz[ver_source, :]
        xyz_target = xyz[ver_target, :]
        graph["source"][i_sedg] = edge_comp[0, i_edg_begin]
        graph["target"][i_sedg] = edge_comp[1, i_edg_begin]
        # ---compute the ratio features---
        # import ipdb
        # ipdb.set_trace()
        graph["se_delta_centroid"][i_sedg, :] = graph["sp_centroids"][com_source, :] - graph["sp_centroids"][com_target,
                                                                                       :]
        graph["se_length_ratio"][i_sedg] = graph["sp_length"][com_source] / (graph["sp_length"][com_target] + 1e-6)
        graph["se_surface_ratio"][i_sedg] = graph["sp_surface"][com_source] / (graph["sp_surface"][com_target] + 1e-6)
        graph["se_volume_ratio"][i_sedg] = graph["sp_volume"][com_source] / (graph["sp_volume"][com_target] + 1e-6)
        graph["se_point_count_ratio"][i_sedg] = graph["sp_point_count"][com_source] / (
                graph["sp_point_count"][com_target] + 1e-6)
        # ---compute the offset set---
        delta = xyz_source - xyz_target
        if len(delta) > 1:
            graph["se_delta_mean"][i_sedg] = np.mean(delta, axis=0)
            graph["se_delta_std"][i_sedg] = np.std(delta, axis=0)
            graph["se_delta_norm"][i_sedg] = np.mean(np.sqrt(np.sum(delta ** 2, axis=1)))
        else:
            graph["se_delta_mean"][i_sedg, :] = delta
            graph["se_delta_std"][i_sedg, :] = [0, 0, 0]
            graph["se_delta_norm"][i_sedg] = np.sqrt(np.sum(delta ** 2))
    return graph



def gen_cutting_gth3(source, target, seg2instlabel, seg2sem):
    """
        v3: _superpointgraph_dmax{}cm_v3.json
        For each edge,
            0 two nodes belong to same semantic class while different instance.
            1 two nodes belong to same instance.
            -100 other wise.

        dmax 30cm may be better since ther will be more connections.
        Online sampling cutting and preserving edges to make a balanced training.
    :param source:
    :param target:
    :param seg2instlabel:
    :param seg2sem
    :return:
        se_same_inst_label: np.array with shape (num_edgs,). 1 means preserve, 0 means cutting, -1 means dont care
    """

    se_same_inst_label = []
    for src, tgt in zip(source, target):
        if seg2instlabel[src] >= 0 and  seg2instlabel[tgt] >= 0:
            if (seg2instlabel[src] == seg2instlabel[tgt]):
                se_same_inst_label.append(1)
            else:
                if seg2sem[src] == seg2sem[tgt]:
                    se_same_inst_label.append(0)
                else:
                    se_same_inst_label.append(-100)
        else:
            se_same_inst_label.append(-100)
    return se_same_inst_label


def gen_scene_spg(pth_path, out_path, d_max=0.05):

    print(pth_path)
    data = torch.load(pth_path)
    if len(data) == 7:  # scannet train_normal_seg val_normal_seg
        xyz, rgb, label, instance_label, normal, color_seg_label, normal_seg_label = data
    elif len(data) == 3:  # scannet test
        xyz, rgb, normal_seg_label = data
        instance_label = None
        normal = None
        label = None
        color_seg_label = None
    else:
        raise KeyError

    if instance_label is not None:
        seg2instlabel = {}
        for inst_label, seg_label in zip(instance_label, normal_seg_label):
            seg2instlabel[seg_label] = inst_label
    else:
        seg2instlabel = None

    if label is not None:
        seg2sem = dict(zip(normal_seg_label, label))
    else:
        seg2sem = None

    components_dic = defaultdict(list)
    for i, seg_id in enumerate(normal_seg_label):
        components_dic[seg_id].append(i)

    components = []
    seg_ids = []
    for seg_id, component in components_dic.items():
        seg_ids.append(seg_id)
        components.append(component)

    normal_seg_label = normal_seg_label.astype('uint32')
    components = np.array(components, dtype='object')

    n_labels = 20
    if label is not None:
        label = label.astype('uint32')
        graph = compute_sp_graph(xyz, d_max=d_max, in_component=normal_seg_label, components=components, seg_ids=seg_ids,
                                 labels=label, n_labels=20)
    else:
        graph = compute_sp_graph(xyz, d_max=d_max, in_component=normal_seg_label, components=components, seg_ids=seg_ids,
                                 labels=[], n_labels=20)
    full_graph = {}
    full_graph['source'] = graph['source'].astype(np.int).squeeze().tolist()
    full_graph['target'] = graph['target'].astype(np.int).squeeze().tolist()

    full_graph['sp_centroids'] = graph['sp_centroids'].tolist()
    full_graph['sp_length'] = graph['sp_length'].tolist()
    full_graph['sp_surface'] = graph['sp_surface'].tolist()
    full_graph['sp_volume'] = graph['sp_volume'].tolist()
    full_graph['sp_point_count'] = graph['sp_point_count'].tolist()
    full_graph['se_delta_mean'] = graph['se_delta_mean'].tolist()
    full_graph['se_delta_std'] = graph['se_delta_std'].tolist()
    full_graph['se_delta_norm'] = graph['se_delta_norm'].tolist()
    full_graph['se_delta_centroid'] = graph['se_delta_centroid'].tolist()
    full_graph['se_length_ratio'] = graph['se_length_ratio'].tolist()
    full_graph['se_surface_ratio'] = graph['se_surface_ratio'].tolist()
    full_graph['se_volume_ratio'] = graph['se_volume_ratio'].tolist()
    full_graph['se_point_count_ratio'] = graph['se_point_count_ratio'].tolist()


    ### v3
    if seg2instlabel is not None:
        full_graph['se_same_inst_label'] = gen_cutting_gth3(full_graph['source'], full_graph['target'], seg2instlabel, seg2sem)


    light_graph = {}
    light_graph['source'] = graph['source'].astype(np.int).squeeze().tolist()
    light_graph['target'] = graph['target'].astype(np.int).squeeze().tolist()

    if seg2instlabel is not None:
        light_graph['se_same_inst_label'] = full_graph['se_same_inst_label']
        graph['se_same_inst_label'] = np.array(full_graph['se_same_inst_label']).astype(np.int)

        se_same_inst_label = np.array(full_graph['se_same_inst_label'])
        print('Total edges:{}, Num of dont care edges: {}, same instance sedges: {}, cross instance edges: {}, ratio: {}'.format(
            len(se_same_inst_label),
            np.sum(se_same_inst_label == -100),
            np.sum(se_same_inst_label == 1),
            np.sum(se_same_inst_label == 0),
            np.sum(se_same_inst_label == 0) / float(np.sum(se_same_inst_label == 0) + np.sum(se_same_inst_label == 1))

        ))

    if out_path is not None:
        with open(out_path, 'w') as f:
            json.dump(full_graph, f)

        if out_path.endswith('.json'):
            with open(out_path.replace('.json', '_light.json'), 'w') as f:
                json.dump(light_graph, f)

        print('saved in {}'.format(out_path))

    del light_graph
    del full_graph
    del seg2instlabel
    del xyz, rgb, label, instance_label, normal_seg_label


def gen_scannet_spgs(scannet_root, out_root, d_max=0.3, processes=32):
    """
    Generate superpoint graph for ScanNet
    :param scannet_root:
    :param out_root:
    :param processes:
    :return:
    """
    pth_paths = glob.glob(os.path.join(scannet_root, '*.pth'))
    if d_max <= 0:
        file_postfix = '_superpointgraph_dmaxinf_v2.json'
    else:
        file_postfix = '_superpointgraph_dmax{}cm_v2.json'.format(int(d_max * 100))

    out_paths = [os.path.join(out_root, os.path.basename(pth_path).replace('.pth', file_postfix)) for pth_path in pth_paths]

    if not os.path.exists(out_root):
        os.makedirs(out_root)

    # gen_scene_spg(pth_paths[0], out_paths[0], d_max)
    pool = multiprocessing.Pool(processes=processes)
    for pth_path, out_path in zip(pth_paths, out_paths):
        pool.apply_async(gen_scene_spg, (pth_path, out_path, d_max))
    pool.close()
    pool.join()




if __name__ == '__main__':
    ### generate superpoint graphs for ScanNet
    scannet_root = 'workspace/PointGroup/dataset/scannetv2/val_normal_seg'
    out_root = 'workspace/PointGroup/dataset/scannetv2/val_normal_seg'
    gen_scannet_spgs(scannet_root, out_root, d_max=-1, processes=32)

    scannet_root = 'workspace/PointGroup/dataset/scannetv2/train_normal_seg'
    out_root = 'workspace/PointGroup/dataset/scannetv2/train_normal_seg'
    gen_scannet_spgs(scannet_root, out_root, d_max=-1, processes=32)

    scannet_root = 'workspace/PointGroup/dataset/scannetv2/test'
    out_root = 'workspace/PointGroup/dataset/scannetv2/test'
    gen_scannet_spgs(scannet_root, out_root, d_max=0.05, processes=32)




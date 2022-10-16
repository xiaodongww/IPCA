# IPCA
## Implicit-Part Based Context Aggregation for Point Cloud Instance Segmentation
Code for the paper **Implicit-Part Based Context Aggregation for Point Cloud Instance Segmentation**, IROS 2022.
![Architecture](./imgs/architecture.jpg)


## Installation
1.  Base environment
Refer to [PointGroup](https://github.com/dvlab-research/PointGroup) for envirinment construction.


2. PointNet++
```bash
cd lib/pointnet2
python setup.py install
```

3. DGL
Install DGL according to https://github.com/dmlc/dgl


## Data Preparation
1. Download ScanNet and unzip it into `dataset/scannetv2/train_normal_seg` and `dataset/scannetv2/val_normal_seg`
2. run the following commands:
```bash
cd dataset/scannetv2/
bash gen_data_with_normal_color_seglabel.sh
```
3. Generate implicit part graph
```bash
cd dataset/scannetv2
python implicit_part_graph.py  # modify dataset root first
```

## Model Description
- Base model: pointgroup/pointgroupleaky.py
- Base model + IPCA: seggroup/segspgsemleaky.py
- Base model + IPCA + SCN: seggroup/segspgsemleakysemscore.py

## Training
Since the training process is time consuming, we initialize our model with a pretrained base model to speed up the training process.
1. Train base PointGropu model
```bash
python train.py --config config/pointgroupleaky_run2_scannet.yaml
```
You can download our trained base model [here]().

2. Train with IPCA module (parameters initialized from base model).
```bash
python train.py --config config/segspgsemleaky_run1_scannet.yaml
```
You can download our trained IPCA model [here]().

3. Train the full model with IPCA and SCN module  (parameters initialized from IPCA model).
```bash
python train.py --config config/segspgsemleakysemscore_run2_scannet.yaml
```



## Evaluation
Test Base model (PointGroup):
```bash
python test.py --config config/pointgroupleaky_run2_scannet.yaml --pretrain pretrained_models/pointgroupleaky_run2_scannet-000000512.pth
```
You should get the following results:
```txt
[2022-10-14 04:27:11,588  INFO  eval.py  line 357  2104]  ################################################################
[2022-10-14 04:27:11,588  INFO  eval.py  line 367  2104]  what           :             AP         AP_50%         AP_25%
[2022-10-14 04:27:11,588  INFO  eval.py  line 368  2104]  ################################################################
[2022-10-14 04:27:11,589  INFO  eval.py  line 381  2104]  cabinet        :          0.364          0.592          0.725
[2022-10-14 04:27:11,589  INFO  eval.py  line 381  2104]  bed            :          0.494          0.756          0.815
[2022-10-14 04:27:11,589  INFO  eval.py  line 381  2104]  chair          :          0.724          0.867          0.906
[2022-10-14 04:27:11,589  INFO  eval.py  line 381  2104]  sofa           :          0.414          0.657          0.816
[2022-10-14 04:27:11,589  INFO  eval.py  line 381  2104]  table          :          0.493          0.700          0.812
[2022-10-14 04:27:11,589  INFO  eval.py  line 381  2104]  door           :          0.271          0.459          0.558
[2022-10-14 04:27:11,589  INFO  eval.py  line 381  2104]  window         :          0.313          0.489          0.688
[2022-10-14 04:27:11,589  INFO  eval.py  line 381  2104]  bookshelf      :          0.234          0.429          0.580
[2022-10-14 04:27:11,589  INFO  eval.py  line 381  2104]  picture        :          0.263          0.422          0.503
[2022-10-14 04:27:11,589  INFO  eval.py  line 381  2104]  counter        :          0.097          0.251          0.583
[2022-10-14 04:27:11,589  INFO  eval.py  line 381  2104]  desk           :          0.181          0.437          0.758
[2022-10-14 04:27:11,590  INFO  eval.py  line 381  2104]  curtain        :          0.303          0.511          0.633
[2022-10-14 04:27:11,590  INFO  eval.py  line 381  2104]  refrigerator   :          0.342          0.458          0.493
[2022-10-14 04:27:11,590  INFO  eval.py  line 381  2104]  shower curtain :          0.492          0.681          0.848
[2022-10-14 04:27:11,590  INFO  eval.py  line 381  2104]  toilet         :          0.854          0.982          0.999
[2022-10-14 04:27:11,590  INFO  eval.py  line 381  2104]  sink           :          0.435          0.723          0.860
[2022-10-14 04:27:11,590  INFO  eval.py  line 381  2104]  bathtub        :          0.647          0.837          0.871
[2022-10-14 04:27:11,590  INFO  eval.py  line 381  2104]  otherfurniture :          0.402          0.595          0.687
[2022-10-14 04:27:11,590  INFO  eval.py  line 390  2104]  ----------------------------------------------------------------
[2022-10-14 04:27:11,590  INFO  eval.py  line 399  2104]  average        :          0.407          0.603          0.730
[2022-10-14 04:27:11,590  INFO  eval.py  line 400  2104]

```


Test IPCA:
```bash
python test_semscore.py --config config/segspgsemleakysemscore_run2_scannet.yaml --pretrain pretrained_models/segspgsemleakysemscore_run2_scannet-000000192.pth
```
You should get the following results:
```txt
[2022-10-14 03:40:16,328  INFO  eval.py  line 357  1467]  ################################################################
[2022-10-14 03:40:16,328  INFO  eval.py  line 367  1467]  what           :             AP         AP_50%         AP_25%
[2022-10-14 03:40:16,328  INFO  eval.py  line 368  1467]  ################################################################
[2022-10-14 03:40:16,328  INFO  eval.py  line 381  1467]  cabinet        :          0.391          0.614          0.765
[2022-10-14 03:40:16,329  INFO  eval.py  line 381  1467]  bed            :          0.550          0.786          0.871
[2022-10-14 03:40:16,329  INFO  eval.py  line 381  1467]  chair          :          0.750          0.887          0.929
[2022-10-14 03:40:16,329  INFO  eval.py  line 381  1467]  sofa           :          0.494          0.740          0.882
[2022-10-14 03:40:16,329  INFO  eval.py  line 381  1467]  table          :          0.555          0.768          0.864
[2022-10-14 03:40:16,329  INFO  eval.py  line 381  1467]  door           :          0.335          0.549          0.647
[2022-10-14 03:40:16,329  INFO  eval.py  line 381  1467]  window         :          0.369          0.549          0.742
[2022-10-14 03:40:16,329  INFO  eval.py  line 381  1467]  bookshelf      :          0.313          0.550          0.673
[2022-10-14 03:40:16,329  INFO  eval.py  line 381  1467]  picture        :          0.493          0.586          0.660
[2022-10-14 03:40:16,329  INFO  eval.py  line 381  1467]  counter        :          0.176          0.360          0.677
[2022-10-14 03:40:16,329  INFO  eval.py  line 381  1467]  desk           :          0.268          0.591          0.835
[2022-10-14 03:40:16,329  INFO  eval.py  line 381  1467]  curtain        :          0.346          0.526          0.659
[2022-10-14 03:40:16,329  INFO  eval.py  line 381  1467]  refrigerator   :          0.543          0.665          0.732
[2022-10-14 03:40:16,330  INFO  eval.py  line 381  1467]  shower curtain :          0.599          0.768          0.867
[2022-10-14 03:40:16,330  INFO  eval.py  line 381  1467]  toilet         :          0.910          0.981          0.981
[2022-10-14 03:40:16,330  INFO  eval.py  line 381  1467]  sink           :          0.551          0.804          0.900
[2022-10-14 03:40:16,330  INFO  eval.py  line 381  1467]  bathtub        :          0.691          0.805          0.870
[2022-10-14 03:40:16,330  INFO  eval.py  line 381  1467]  otherfurniture :          0.485          0.648          0.740
[2022-10-14 03:40:16,330  INFO  eval.py  line 390  1467]  ----------------------------------------------------------------
[2022-10-14 03:40:16,330  INFO  eval.py  line 399  1467]  average        :          0.490          0.676          0.794
[2022-10-14 03:40:16,330  INFO  eval.py  line 400  1467]

```

## Citation
If you find this work useful in your research, please cite:
```txt
@inproceedings{iros2022/ipca,
  author    = {Xiaodong Wu and
               Ruiping Wang and
               Xilin Chen},
  title     = {Implicit-Part Based Context Aggregation for Point Cloud Instance Segmentation},
  booktitle = {{IEEE/RSJ} International Conference on Intelligent Robots and Systems,
               {IROS} 2022},
  publisher = {{IEEE}},
  year      = {2022},
}
```


## Acknowledgement
This repo is built upon [PointGroup](https://github.com/dvlab-research/PointGroup), [Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch), [superpoint_graph
](https://github.com/loicland/superpoint_graph).

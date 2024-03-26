# @Copyright (c) 2019-2022 haomo.ai, Inc. All Rights Reserved
import shapely
from shapely.geometry import Polygon,MultiPoint 
import copy
import rtree
import torch
import numpy as np
import cv2
# from tools.gaussian_target import *
from mmcv.ops import batched_nms
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

# copy from mmdet2.x: mmdet/core/anchor/point_generator.py
class MlvlPointGenerator:
    """Standard points generator for multi-level (Mlvl) feature maps in 2D
    points-based detectors.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        offset (float): The offset of points, the value is normalized with
            corresponding stride. Defaults to 0.5.
    """

    def __init__(self, strides, offset=0.5):
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    @property
    def num_base_priors(self):
        """list[int]: The number of priors (points) at a point
        on the feature grid"""
        return [1 for _ in range(len(self.strides))]

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_priors(self, featmap_sizes, device='cuda', with_stride=False):
        """Generate grid points of multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            device (str): The device where the anchors will be put on.
            with_stride (bool): Whether to concatenate the stride to
                the last dimension of points.

        Return:
            list[torch.Tensor]: Points of  multiple feature levels.
            The sizes of each tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(
                featmap_sizes[i],
                level_idx=i,
                device=device,
                with_stride=with_stride)
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(self,
                                 featmap_size,
                                 level_idx,
                                 device='cuda',
                                 with_stride=False):
        """Generate grid Points of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps, arrange as
                (h, w).
            level_idx (int): The index of corresponding feature map level.
            device (str, optional): The device the tensor will be put on.
                Defaults to 'cuda'.
            with_stride (bool): Concatenate the stride to the last dimension
                of points.

        Return:
            Tensor: Points of single feature levels.
            The shape of tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0., feat_w, device=device) +
                   self.offset) * stride_w
        shift_y = (torch.arange(0., feat_h, device=device) +
                   self.offset) * stride_h
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            stride_w = shift_xx.new_full((len(shift_xx), ), stride_w)
            stride_h = shift_xx.new_full((len(shift_yy), ), stride_h)
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h],
                                 dim=-1)
        all_points = shifts.to(device)
        return all_points

    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
        """Generate valid flags of points of multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            pad_shape (tuple(int)): The padded shape of the image,
                 arrange as (h, w).
            device (str): The device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of points of multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            point_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / point_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 device='cuda'):
        """Generate the valid flags of points of a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps, arrange as
                as (h, w).
            valid_size (tuple[int]): The valid size of the feature maps.
                The size arrange as as (h, w).
            device (str, optional): The device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each points in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid

    def sparse_priors(self,
                      prior_idxs,
                      featmap_size,
                      level_idx,
                      dtype=torch.float32,
                      device='cuda'):
        """Generate sparse points according to the ``prior_idxs``.

        Args:
            prior_idxs (Tensor): The index of corresponding anchors
                in the feature map.
            featmap_size (tuple[int]): feature map size arrange as (w, h).
            level_idx (int): The level index of corresponding feature
                map.
            dtype (obj:`torch.dtype`): Date type of points. Defaults to
                ``torch.float32``.
            device (obj:`torch.device`): The device where the points is
                located.
        Returns:
            Tensor: Anchor with shape (N, 2), N should be equal to
            the length of ``prior_idxs``. And last dimension
            2 represent (coord_x, coord_y).
        """
        height, width = featmap_size
        x = (prior_idxs % width + self.offset) * self.strides[level_idx][0]
        y = ((prior_idxs // width) % height +
             self.offset) * self.strides[level_idx][1]
        prioris = torch.stack([x, y], 1).to(dtype)
        prioris = prioris.to(device)
        return prioris

# copy from mmdet2.x: mmdet/haomo/models/heads/yolox_base_head_v3.py
def _bbox_decode(priors, bbox_preds):
    # bbox_preds = F.sigmoid(bbox_preds)
    # xys = (2 * bbox_preds[..., :2] - 0.5) + priors[:, :2]
    # whs = ((12 * bbox_preds[..., 2:]) ** 2) * priors[:, 2:]
    xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
    whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

    tl_x = (xys[..., 0] - whs[..., 0] / 2)
    tl_y = (xys[..., 1] - whs[..., 1] / 2)
    br_x = (xys[..., 0] + whs[..., 0] / 2)
    br_y = (xys[..., 1] + whs[..., 1] / 2)

    decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
    return decoded_bboxes


def decode_prob_depth(depth_cls_preds, depth_range, depth_unit,
                          division, num_depth_cls, camx=False, camz=False):
    """Decode probabilistic depth map.

    Args:
        depth_cls_preds (torch.Tensor): Depth probabilistic map in shape
            [..., self.num_depth_cls] (raw output before softmax).
        depth_range (tuple[float]): Range of depth estimation.
        depth_unit (int): Unit of depth range division.
        division (str): Depth division method. Options include 'uniform',
            'linear', 'log', 'loguniform'.
        num_depth_cls (int): Number of depth classes.

    Returns:
        torch.Tensor: Decoded probabilistic depth estimation.
    """
    if division == 'uniform':
        # depth_multiplier = depth_unit * \
        #     depth_cls_preds.new_tensor(
        #         list(range(num_depth_cls))).reshape([1, -1])
        # if camz:
        #     depth_multiplier = depth_multiplier - 10
        # elif camx:
        #     depth_multiplier = depth_multiplier - 100 
        # prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) *
        #                     depth_multiplier).sum(dim=-1)
        # return prob_depth_preds
        
        # torch.Size([1, 14])
        depth_multiplier = depth_cls_preds.new_tensor([i for i in range(depth_range[0],depth_range[1]+10,10)]).reshape([1,-1])
        if camz:
            # 最后一个距离值+5m
            depth_multiplier[...,-1] = depth_multiplier[...,-1]+5
        # torch.Size([1, 12096, 14]) * torch.Size([1, 14])--> torch.Size([1, 12096])
        prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) *
                            depth_multiplier).sum(dim=-1)
        return prob_depth_preds
    elif division == 'linear':
        split_pts = depth_cls_preds.new_tensor(list(
            range(num_depth_cls))).reshape([1, -1])
        depth_multiplier = depth_range[0] + (
            depth_range[1] - depth_range[0]) / \
            (num_depth_cls * (num_depth_cls - 1)) * \
            (split_pts * (split_pts+1))
        prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) *
                            depth_multiplier).sum(dim=-1)
        return prob_depth_preds
    elif division == 'log':
        split_pts = depth_cls_preds.new_tensor(list(
            range(num_depth_cls))).reshape([1, -1])
        start = max(depth_range[0], 1)
        end = depth_range[1]
        depth_multiplier = (np.log(start) +
                            split_pts * np.log(end / start) /
                            (num_depth_cls - 1)).exp()
        prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) *
                            depth_multiplier).sum(dim=-1)
        return prob_depth_preds
    elif division == 'loguniform':
        split_pts = depth_cls_preds.new_tensor(list(
            range(num_depth_cls))).reshape([1, -1])
        start = max(depth_range[0], 1)
        end = depth_range[1]
        log_multiplier = np.log(start) + \
            split_pts * np.log(end / start) / (num_depth_cls - 1)
        prob_depth_preds = (F.softmax(depth_cls_preds.clone(), dim=-1) *
                            log_multiplier).sum(dim=-1).exp()
        return prob_depth_preds
    else:
        raise NotImplementedError


#############################################################################################################################
    
# modified refer: mmdet/haomo/models/heads/vehicle_yolox_head_v3.py的get_bbox()
def decode_det_results(
                predicts,
                strides=[8, 16, 32],
                bbox_score_thr=0.45,
                bbox_nms=dict(type='nms', iou_threshold=0.65),
                img_metas=None,
                rescale=True,
                task_type = "vehicle",
                use_depth_classifier = False,
            ):
    """
    cls_scores, list, 数量为fpn输出的个数, 其中一个特征图的shape, 例如:torch.Size([1, 3, 72, 128])
    bbox_preds,4
    objectnesses, 1 
    attrs_preds, 16 : self.occlusion_dims(3) + self.truncation_dims(2)
        + self.crowding_dims(3) + direction(8)
    fake3ds_preds, 12 :  4(汽车图中3D bbox下平面的左前,左后,右前,右后，四个点是否可见）
        +4*2(上述四个点在图像坐标系中的坐标), >>>>转onnx时不输出,见forward_dummy
    union3ds_preds, 34 :  size_dims(height, length, width)3+【方体中心centerness2+depths_dims1】或【positions_dims(x,y,z)3】
        +yaw_dims(yaw_sin', 'yaw_cos')2+flag_120(2)+corners_dims(24) 上述x,y,z和yaw是在车体坐标系下的值, >>>>转onnx时只输出前topk个
    depth_cls_preds, 27 :  复用union3d的head输出的feats, 
        预测器设置depth_classifier_dims=camx_bins(横向13,-60~60m)+ depth_bins(纵向14,-10~120m) 
    weight_preds, 空
    dir_cls_preds, 空
    extra_cls_preds, 空
    struct_preds, 9: 车灯状态分类，左转灯(unknown, on, off),右转灯(unknown, on, off),刹车灯(unknown, on, off) 3+3+3=9
    block_depth_preds 空

    转onnx时输出:
    39=vehicle_pred_maps_level = torch.cat([bbox_maps[i]4, objectness_maps[i]1, cls_maps[i]3, 
        union3d_maps[i][:, :10, :, :]10, depth_cls_maps[i]27, attrs_maps[i]16, struct_maps[i]]9, dim=1)
    
    """ 
    prior_generator = MlvlPointGenerator(strides, offset=0)

    cls_scores = None
    bbox_preds = None
    objectnesses = None
    attrs_preds = None
    fake3ds_preds = None
    union3ds_preds = None
    depth_cls_preds = None
    weight_preds = None
    dir_cls_preds = None
    extra_cls_preds = None
    struct_preds = None
    block_depth_preds = None

    bbox_preds = [tmp[:, 0:4] for tmp in predicts]
    objectnesses = [tmp[:, 4:5] for tmp in predicts]

    num_classes=0
    union3ds_dims=0
    attrs_dims = 0

    if task_type=="vehicle":
        cls_scores = [tmp[:, 5:8] for tmp in predicts]
        num_classes = cls_scores[0].shape[1]
        
        union3ds_preds = [tmp[:, 8:18] for tmp in predicts]
        union3ds_dims = union3ds_preds[0].shape[1]
        depth_cls_preds = [tmp[:, 18:45] for tmp in predicts]
        attrs_preds = [tmp[:, 45:61] for tmp in predicts]
        attrs_dims = attrs_preds[0].shape[1]
        # 车灯状态属性
        struct_preds = [tmp[:, 61:70] for tmp in predicts]

    if task_type=="vru":
        cls_scores = [tmp[:, 5:9] for tmp in predicts]
        num_classes = cls_scores[0].shape[1]

        union3ds_preds = [tmp[:, 9:19] for tmp in predicts]
        union3ds_dims = union3ds_preds[0].shape[1]
        depth_cls_preds = [tmp[:, 19:46] for tmp in predicts]
        attrs_preds = [tmp[:, 46:56] for tmp in predicts]
        attrs_dims = attrs_preds[0].shape[1]

    if task_type=="static":
        cls_scores = [tmp[:, 5:7] for tmp in predicts]
        num_classes = cls_scores[0].shape[1]
        union3ds_preds = [tmp[:, 7:13] for tmp in predicts]
        union3ds_dims = union3ds_preds[0].shape[1]

    assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
    
    #scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]

    #batch size
    num_imgs = len(img_metas)
    #[torch.Size([72, 128]), torch.Size([36, 64]), torch.Size([18, 32])]
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
    #torch.Size([9216, 4]) torch.Size([2304, 4]) torch.Size([576, 4])
    # 每个特征图上的点位置变为原图尺度下的grid point点位置，一一对应，(coord_x, coord_y, stride_w, stride_h)
    mlvl_priors = prior_generator.grid_priors(
        featmap_sizes, cls_scores[0].device, with_stride=True)

    # flatten cls_scores, bbox_preds and objectness, 预测维度放最后，HW合并
    cls_out_channels = cls_scores[0].shape[1]
    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, cls_out_channels)
        for cls_score in cls_scores
    ]
    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
        for bbox_pred in bbox_preds
    ]
    flatten_objectness = [
        objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
        for objectness in objectnesses
    ]

    #第dim=1的维度都变为12096
    flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
    flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
    flatten_priors = torch.cat(mlvl_priors)

    # 预测的flatten_bbox_preds各特征图尺度上的xyhw通过flatten_priors，映射为输入backbone图尺度的xyxy形式bbox坐标
    # 返回特征图上每个点对应的预测结果torch.Size([1, 12096, 4])
    flatten_bboxes = _bbox_decode(flatten_priors, flatten_bbox_preds)
    
    if rescale:
        flatten_bboxes[..., :4] /= flatten_bboxes.new_tensor(
            scale_factors).unsqueeze(1)#scale_factors[arr[4]]变为torch.Size([1, 1, 4])

    # attrs
    # 特征图上每个点位置预测的各属性概率值,16=self.occlusion_dims(3) + 
    # self.truncation_dims(2) + self.crowding_dims(3) + direction(8)
    if attrs_preds:
        # print(attrs_preds[0].shape, attrs_dims)
        flatten_attrs_preds = [
            attrs_pred.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, attrs_dims)
            for attrs_pred in attrs_preds]
        # torch.Size([1, 12096, 16])
        flatten_attrs_preds = torch.cat( flatten_attrs_preds, dim=1).sigmoid()

    # # fake3ds
    # # 12 :  4 (汽车图中3D bbox下平面的左前,左后,右前,右后，四个点是否可见）+4*2(上述四个点在图像坐标系中的坐标)
    # if fake3ds_preds:
    #     flatten_fake3ds_preds = [
    #         fake3ds_pred.permute(0, 2, 3, 1).reshape(
    #             num_imgs, -1, fake3ds_dims)
    #         for fake3ds_pred in fake3ds_preds]
    #     flatten_fake3ds_preds = torch.cat(flatten_fake3ds_preds, dim=1)
    #     flatten_fake3ds_flags = flatten_fake3ds_preds[..., :self.fake3d_flag_dims].sigmoid()
    #     flatten_fake3ds_cords = flatten_fake3ds_preds[..., self.fake3d_flag_dims:]
    #     # torch.Size([1, 12096, 8])
    #     flatten_fake3ds_cords = self._fake3d_decode(flatten_priors, flatten_fake3ds_cords)
    #     if rescale:
    #         flatten_fake3ds_cords[..., :] /= flatten_fake3ds_cords.new_tensor(
    #             scale_factors).repeat(1, 2).unsqueeze(1)#scale_factors[arr[4]]变为torch.Size([1, 8])再变为torch.Size([1, 1, 8])
    
    # union3ds
    # 34 :  size_dims(height, length, width)3+【方体中心centerness2+depths_dims1】或【positions_dims(x,y,z)3】
    # +yaw_dims(yaw_sin', 'yaw_cos')2+flag_120(2)+corners_dims(24) 上述x,y,z和yaw是在车体坐标系下的值
    flatten_120_flag_preds_score = None
    flatten_rotations_preds = None
    if union3ds_preds:
        # self.visualization(union3ds_preds[0], img_metas, "/mnt/share_disk/zhangchen/data/visualization/v3.3_alldata/feature/union3d/small")
        # self.visualization(union3ds_preds[1], img_metas, "/mnt/share_disk/zhangchen/data/visualization/v3.3_alldata/feature/union3d/middle")
        # self.visualization(union3ds_preds[2], img_metas, "/mnt/share_disk/zhangchen/data/visualization/v3.3_alldata/feature/union3d/large")
        class_agnostic = True
        if not class_agnostic:
            flatten_union3ds_preds = [union3ds_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, union3ds_dims * num_classes) for union3ds_pred in union3ds_preds]
        else:
            flatten_union3ds_preds = [union3ds_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, union3ds_dims) for union3ds_pred in union3ds_preds]
        
        # torch.Size([1, 12096, 34])
        flatten_union3ds_preds = torch.cat(flatten_union3ds_preds, dim=1)

        # if not class_agnostic:
        #     assert len(flatten_union3ds_preds.shape) == 3
        #     b1, n2, c3 = flatten_union3ds_preds.shape
        #     flatten_union3ds_preds = flatten_union3ds_preds.view(b1, n2, -1, num_classes)

        #     _, classification_results = torch.max(flatten_cls_scores, 2)
        #     flatten_union3ds_preds = torch.gather(flatten_union3ds_preds, dim=3,
        #                                            index=classification_results.unsqueeze(-1).unsqueeze(-1).repeat(
        #                                                1, 1, c3 // self.num_classes, 1)).squeeze(-1)
        
        size_dims = 0
        flatten_dimensions_preds = flatten_union3ds_preds[..., 0:3]
        size_dims = 3
        flatten_positions_preds = flatten_union3ds_preds[..., 3:6]
        size_dims = 6
        
        if use_depth_classifier==False:
            # 此处预测的位置对应自车坐标系的z,x,y,注意推理或测试时需转换成xyz的自车坐标系预测顺序
            flatten_camx_preds = flatten_positions_preds[...,0] #对应原始标注的自车坐标系的y
            flatten_camy_preds = flatten_positions_preds[...,1] #对应原始标注的自车坐标系的z
            flatten_camz_preds = flatten_positions_preds[...,2] #对应原始标注的自车坐标系的x
            # 在此处为flatten_positions_preds修正对应原始标注的车体坐标系的真实xyz标注顺序
            flatten_positions_preds = torch.stack((flatten_camz_preds, flatten_camx_preds,flatten_camy_preds),dim=2)

        # 增强flatten_positions_preds位置预测
        if use_depth_classifier==True:
            num_depth_cls = 14
            num_camx_cls = 13
            depth_range=(-10, 120)
            depth_unit=10
            division='uniform'
            depth_bins=14
            camx_range=(-60, 60)
            camx_unit=10
            camx_bins=13
            corners_dims = 24
            original_x=0
            original_y=0
            sig_alpha = 0.8520543575286865
            camx_alpha = 0.7439162135124207    
            print(">>>> use_depth_classifier=True!!!!")  
            print(">>>> sig_alpha={}".format(sig_alpha))    
            print(">>>> camx_alpha={}".format(camx_alpha))     

            # depth:障碍物所处的相机位置坐标
            flatten_camx_preds = flatten_positions_preds[...,0]
            flatten_camy_preds = flatten_positions_preds[...,1]
            flatten_camz_preds = flatten_positions_preds[...,2]

            # depth_cls 
            # 复用union3d的head输出的feats,
            # 预测器设置depth_classifier_dims=camx_bins(横向13,-60-60m)+ depth_bins(纵向14,-10-120m)
            # 纵向距离 14个bin 
            flatten_depths_cls_preds =  [
                depth_cls_pred[:, :num_depth_cls, :].permute(0, 2, 3, 1).reshape(
                    num_imgs, -1, num_depth_cls)
                for depth_cls_pred in depth_cls_preds]
            # torch.Size([1, 12096, 14])
            flatten_depths_cls_preds = torch.cat(flatten_depths_cls_preds, dim=1)
            
            # 横向距离13个bin 
            flatten_camxs_cls_preds =  [
                depth_cls_pred[:, num_depth_cls:, :].permute(0, 2, 3, 1).reshape(
                    num_imgs, -1, num_camx_cls)
                for depth_cls_pred in depth_cls_preds]
            # torch.Size([1, 12096, 13])
            flatten_camxs_cls_preds = torch.cat(flatten_camxs_cls_preds, dim=1)

            # if self.use_fusion_weight:
            #     #fusion weight
            #     flatten_weights_preds =  [
            #         weight_pred.permute(0, 2, 3, 1).reshape(
            #             num_imgs, -1, self.weight_dims)
            #         for weight_pred in weight_preds]
            #     flatten_weights_preds = torch.cat(flatten_weights_preds, dim=1)
            # depth_range=(-10, 120), depth_unit=10 division='uniform', 输出torch.Size([1, 12096])
            
            ### 得到预测的3d目标中心位置信息 flatten_positions_preds
            pos_prob_camz_preds = decode_prob_depth(
            flatten_depths_cls_preds, depth_range, depth_unit,
            division, num_depth_cls, camz=True)
            flatten_camzs_preds = sig_alpha * flatten_camz_preds + (1 - sig_alpha) * pos_prob_camz_preds   

            pos_prob_camx_preds = decode_prob_depth(
            flatten_camxs_cls_preds, camx_range, camx_unit,
            division, num_camx_cls, camx=True)
            flatten_camxs_preds = camx_alpha * flatten_camx_preds + (1 - camx_alpha) * pos_prob_camx_preds

            # flatten_positions_preds = torch.stack((flatten_camxs_preds,flatten_camy_preds,flatten_camzs_preds),dim=2)
            #因训练时交换了xyz的顺序位置，此处是调换过来顺序，train(1,2,0) -- test(2,0,1) 
            flatten_positions_preds = torch.stack((flatten_camzs_preds, flatten_camxs_preds,flatten_camy_preds),dim=2)

        if size_dims<union3ds_dims:
            # yaw角预测，torch.Size([1, 12096, 1]), 
            flatten_rotations_preds = torch.atan2(flatten_union3ds_preds[..., 6:7] / 5.0 - 1.0, flatten_union3ds_preds[..., 7:8] / 5.0 - 1.0)
            size_dims = size_dims + 2
            # 增强yaw角预测     
            # if dir_cls_preds:
            #     # 方向预测，复用union3d的head输出的feats，预测器设置dir_bins=2，dir_bins为预测器输出channel维度
            #     flatten_dir_cls_preds = [
            #         dir_cls_pred[:,:self.dir_bins,:].permute(0, 2, 3, 1).reshape(num_imgs, -1, self.dir_bins) for dir_cls_pred in dir_cls_preds
            #     ]
            #     flatten_dir_cls_preds = torch.cat(flatten_dir_cls_preds, dim=1)
            #     flatten_dir_cls_preds = torch.max(flatten_dir_cls_preds, dim=-1)[1].unsqueeze(-1)
            #     #decode_yaw
            #     if flatten_rotations_preds.shape[0]>0:
            #         flatten_rotations_preds = self.decode_yaw(flatten_rotations_preds, flatten_dir_cls_preds, self.dir_offset)
            # print("@", flatten_rotations_preds.shape, flatten_rotations_preds)

            if size_dims<union3ds_dims:
                flatten_120_flag_preds = flatten_union3ds_preds[..., 8:10]
                # torch.Size([1, 12096])
                flatten_120_flag_preds = torch.max(flatten_120_flag_preds, dim=-1)[1]
                flatten_120_flag_preds_score = flatten_union3ds_preds[...,  8:10].sigmoid()
                # print(flatten_120_flag_preds_score)

            # onnx模型不输出
            # if self.extra_3d_points:
            #     start_dims += self.flag_120
            #     end_dims += self.corners_dims
            #     # torch.Size([1, 12096, 24])
            #     flatten_extra_3d_points_preds = flatten_union3ds_preds[..., start_dims:end_dims]

    if struct_preds:
        # 车灯状态分类，左转灯(unknown, on, off),右转灯(unknown, on, off),刹车灯(unknown, on, off) 3+3+3=9
        flatten_struct_preds = [struct_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 9) for struct_pred in struct_preds]
        flatten_struct_preds = torch.cat(flatten_struct_preds, dim=1).sigmoid()

    result_list = []
    for img_id in range(len(img_metas)):
        # torch.Size([12096, 3])
        cls_scores = flatten_cls_scores[img_id]
        # torch.Size([12096])
        score_factor = flatten_objectness[img_id]
        # torch.Size([12096, 4])
        bboxes = flatten_bboxes[img_id]

        # torch.Size([12096]), torch.Size([12096])
        max_scores, labels = torch.max(cls_scores, 1)
        # print(max_scores)
        valid_mask = score_factor * max_scores >= bbox_score_thr
        # if self.split_120:
        #     flag_120_mask = (flatten_120_flag_preds[img_id]==1)
        #     valid_mask = valid_mask & flag_120_mask

        # torch.Size([60, 4])
        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask] * score_factor[valid_mask]
        labels = labels[valid_mask]
        result = []
        if labels.numel() == 0:
            result.append(bboxes)
            result.append(scores)
            if attrs_preds:
                result.append(flatten_attrs_preds[img_id][valid_mask])
            # if self.pred_fake3d:
            #     result.append(flatten_fake3ds_flags[img_id][valid_mask])
            #     result.append(flatten_fake3ds_cords[img_id][valid_mask])
            if union3ds_preds:
                result.append(flatten_dimensions_preds[img_id][valid_mask])
                # if self.use_uvd:
                #     if self.use_uvd_position_consistency:
                #         result.append(flatten_positions_preds[img_id][valid_mask])
                #     # result.append(flatten_centerness_preds[img_id][valid_mask])
                #     # result.append(flatten_depths_preds[img_id][valid_mask])
                # else:
                result.append(flatten_positions_preds[img_id][valid_mask])

                if flatten_rotations_preds is not None:
                    # yaw角预测，推理时候yaw = arctran(预测的坐标y, 预测的坐标x) + arctran((output1 / 5 - 1), 
                    #     (output2 / 5 - 1))，因为预测的角度值 = yaw - arctran(position_y, position_x)  
                    result.append(torch.atan2(flatten_positions_preds[img_id][valid_mask][...,1], 
                                                flatten_positions_preds[img_id][valid_mask][...,0]).unsqueeze(1)+ 
                                                flatten_rotations_preds[img_id][valid_mask])
            
                if flatten_120_flag_preds_score is not None:
                    result.append(flatten_120_flag_preds_score[img_id][valid_mask])
                
                # if self.extra_3d_points:
                #     result.append(flatten_extra_3d_points_preds[img_id][valid_mask])
            
            if struct_preds:
                result.append(flatten_struct_preds[img_id][valid_mask])

            result = torch.cat(result, dim=-1)
            result_list.append([result, labels])

        else:
            # Some type of nms would reweight the score, such as SoftNMS
            dets, keep = batched_nms(bboxes, scores, labels, bbox_nms)
            result.append(dets)#13*5
            if attrs_preds:
                result.append(flatten_attrs_preds[img_id][valid_mask][keep])#13*16
                
            # if self.pred_fake3d:
            #     result.append(flatten_fake3ds_flags[img_id][valid_mask][keep])#13*4
            #     result.append(flatten_fake3ds_cords[img_id][valid_mask][keep])#13*8
            
            if union3ds_preds:
                result.append(flatten_dimensions_preds[img_id][valid_mask][keep])#13*3
                # if self.use_uvd:
                #     if self.use_uvd_position_consistency:
                #         result.append(flatten_positions_preds[img_id][valid_mask][keep])
                #     # result.append(flatten_centerness_preds[img_id][valid_mask][keep])
                #     # result.append(flatten_depths_preds[img_id][valid_mask][keep])
                # else:
                result.append(flatten_positions_preds[img_id][valid_mask][keep])#13*3
                
                if flatten_rotations_preds is not None:
                    # yaw角预测，推理时候yaw = arctran(预测的坐标y, 预测的坐标x) + arctran((output1 / 5 - 1), 
                    #     (output2 / 5 - 1))，因为预测的角度值 = yaw - arctran(position_y, position_x)
                    result.append(torch.atan2(flatten_positions_preds[img_id][valid_mask][keep][...,1], 
                                                flatten_positions_preds[img_id][valid_mask][keep][...,0]).unsqueeze(1)+ 
                                                flatten_rotations_preds[img_id][valid_mask][keep])#13*1
                # camera coordinate
                # result.append(torch.atan2(flatten_positions_preds[img_id][valid_mask][keep][...,0], 
                # flatten_positions_preds[img_id][valid_mask][keep][...,2]).unsqueeze(1)+ flatten_rotations_preds[img_id][valid_mask][keep])
                # result.append(flatten_rotations_preds[img_id][valid_mask][keep])

                if flatten_120_flag_preds_score is not None:
                    result.append(flatten_120_flag_preds_score[img_id][valid_mask][keep])#13*2
                    # print("@", result[-1].shape)
                
                # if self.extra_3d_points:
                #     result.append(flatten_extra_3d_points_preds[img_id][valid_mask][keep])#13*24
            
            if struct_preds:
                result.append(flatten_struct_preds[img_id][valid_mask][keep])#13*9

            # torch-vehicle模型返回:75 = (bbox+score)5+(遮挡，方向等属性)16+12(左前,左后,右前,右后四个点是否可见4+四个点坐标8)
            #                      +(h,w,z)3+(xyz)3+(yaw)1+(是否120m内)2+(8个corners)24+(左转灯/右转灯/刹车灯)9
            # onnx-vehicle模型返回:39 = (bbox+score)5+(遮挡，方向等属性)16+(h,w,z)3+(xyz)3+(yaw)1+(是否120m内)2+(左转灯/右转灯/刹车灯)9
            # onnx-vru模型返回:24 = (bbox+score)5+(遮挡，方向等属性)10+(h,w,z)3+(xyz)3+(yaw)1+(是否120m内)2
            # onnx-static模型返回:11 = (bbox+score)5+(h,w,z)3+(xyz)3
            result = torch.cat(result, dim=-1)
            # labels表示检测的目标类别
            result_list.append([result, labels[keep]])
    
    # 把 decode_det_results() 预测的类别相同的实例预测list放到一个list里, 每个类别都会返回一个矩阵，即使预测为空也返回类似0*75的矩阵
    def bbox2result(bboxes, labels, num_classes, flag=False):

        shape = bboxes.shape[-1]
        if bboxes.shape[0] == 0:
            return [np.zeros((0, shape), dtype=np.float32) for i in range(num_classes)]
        else:
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
            return [bboxes[labels == i, :] for i in range(num_classes)]

    results = [bbox2result(det_bboxes, det_labels, num_classes)
               for det_bboxes, det_labels in result_list]

    # print("\n>>>>>>>>>task type = {}, results shape={}, ".format(task_type,  results[0][0].shape))
    return results





# modified refer: mmdet/haomo/models/heads/vehicle_yolox_head_v3.py的get_bbox()
def decode_det_results_toy(
                predicts,
                strides=[8, 16, 32],
                bbox_score_thr=0.45,
                bbox_nms=dict(type='nms', iou_threshold=0.65),
                img_metas=None,
                rescale=True,
                task_type = "vehicle",
                use_depth_classifier = False,
            ):
    """
    cls_scores, list, 数量为fpn输出的个数, 其中一个特征图的shape, 例如:torch.Size([1, 3, 72, 128])
    bbox_preds,4
    objectnesses, 1 
    union3ds_preds, 8 :  size_dims(height, length, width)3+【方体中心centerness2+depths_dims1】或【positions_dims(x,y,z)3】+yaw_dims(yaw_sin', 'yaw_cos')
    
    转onnx时输出: 16=vehicle_pred_maps_level = torch.cat([bbox_maps[i]4, objectness_maps[i]1, cls_maps[i]3, union3d_maps[i]8, dim=1)
    
    """ 
    prior_generator = MlvlPointGenerator(strides, offset=0)

    cls_scores = None
    bbox_preds = None
    objectnesses = None
    attrs_preds = None
    fake3ds_preds = None
    union3ds_preds = None
    depth_cls_preds = None
    weight_preds = None
    dir_cls_preds = None
    extra_cls_preds = None
    struct_preds = None
    block_depth_preds = None

    bbox_preds = [tmp[:, 0:4] for tmp in predicts]
    objectnesses = [tmp[:, 4:5] for tmp in predicts]

    num_classes=0
    union3ds_dims=0
    attrs_dims = 0

    if task_type=="vehicle":
        cls_scores = [tmp[:, 5:8] for tmp in predicts]
        num_classes = cls_scores[0].shape[1]
        
        union3ds_preds = [tmp[:, 8:16] for tmp in predicts]
        union3ds_dims = union3ds_preds[0].shape[1]

    if task_type=="vru":
        cls_scores = [tmp[:, 5:9] for tmp in predicts]
        num_classes = cls_scores[0].shape[1]

        union3ds_preds = [tmp[:, 9:17] for tmp in predicts]
        union3ds_dims = union3ds_preds[0].shape[1]

    # if task_type=="static":
    #     cls_scores = [tmp[:, 5:7] for tmp in predicts]
    #     num_classes = cls_scores[0].shape[1]
    #     union3ds_preds = [tmp[:, 7:13] for tmp in predicts]
    #     union3ds_dims = union3ds_preds[0].shape[1]

    assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
    
    #scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]

    #batch size
    num_imgs = len(img_metas)
    #[torch.Size([72, 128]), torch.Size([36, 64]), torch.Size([18, 32])]
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
    #torch.Size([9216, 4]) torch.Size([2304, 4]) torch.Size([576, 4])
    # 每个特征图上的点位置变为原图尺度下的grid point点位置，一一对应，(coord_x, coord_y, stride_w, stride_h)
    mlvl_priors = prior_generator.grid_priors(
        featmap_sizes, cls_scores[0].device, with_stride=True)

    # flatten cls_scores, bbox_preds and objectness, 预测维度放最后，HW合并
    cls_out_channels = cls_scores[0].shape[1]
    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, cls_out_channels)
        for cls_score in cls_scores
    ]
    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
        for bbox_pred in bbox_preds
    ]
    flatten_objectness = [
        objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
        for objectness in objectnesses
    ]

    #第dim=1的维度都变为12096
    flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
    flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
    flatten_priors = torch.cat(mlvl_priors)

    # 预测的flatten_bbox_preds各特征图尺度上的xyhw通过flatten_priors，映射为输入backbone图尺度的xyxy形式bbox坐标
    # 返回特征图上每个点对应的预测结果torch.Size([1, 12096, 4])
    flatten_bboxes = _bbox_decode(flatten_priors, flatten_bbox_preds)
    
    if rescale:
        flatten_bboxes[..., :4] /= flatten_bboxes.new_tensor(
            scale_factors).unsqueeze(1)#scale_factors[arr[4]]变为torch.Size([1, 1, 4])

    # attrs
    # 特征图上每个点位置预测的各属性概率值,16=self.occlusion_dims(3) + 
    # self.truncation_dims(2) + self.crowding_dims(3) + direction(8)
    if attrs_preds:
        # print(attrs_preds[0].shape, attrs_dims)
        flatten_attrs_preds = [
            attrs_pred.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, attrs_dims)
            for attrs_pred in attrs_preds]
        # torch.Size([1, 12096, 16])
        flatten_attrs_preds = torch.cat( flatten_attrs_preds, dim=1).sigmoid()

    # # fake3ds
    # # 12 :  4 (汽车图中3D bbox下平面的左前,左后,右前,右后，四个点是否可见）+4*2(上述四个点在图像坐标系中的坐标)
    # if fake3ds_preds:
    #     flatten_fake3ds_preds = [
    #         fake3ds_pred.permute(0, 2, 3, 1).reshape(
    #             num_imgs, -1, fake3ds_dims)
    #         for fake3ds_pred in fake3ds_preds]
    #     flatten_fake3ds_preds = torch.cat(flatten_fake3ds_preds, dim=1)
    #     flatten_fake3ds_flags = flatten_fake3ds_preds[..., :self.fake3d_flag_dims].sigmoid()
    #     flatten_fake3ds_cords = flatten_fake3ds_preds[..., self.fake3d_flag_dims:]
    #     # torch.Size([1, 12096, 8])
    #     flatten_fake3ds_cords = self._fake3d_decode(flatten_priors, flatten_fake3ds_cords)
    #     if rescale:
    #         flatten_fake3ds_cords[..., :] /= flatten_fake3ds_cords.new_tensor(
    #             scale_factors).repeat(1, 2).unsqueeze(1)#scale_factors[arr[4]]变为torch.Size([1, 8])再变为torch.Size([1, 1, 8])
    
    # union3ds
    # 34 :  size_dims(height, length, width)3+【方体中心centerness2+depths_dims1】或【positions_dims(x,y,z)3】
    # +yaw_dims(yaw_sin', 'yaw_cos')2+flag_120(2)+corners_dims(24) 上述x,y,z和yaw是在车体坐标系下的值
    flatten_120_flag_preds_score = None
    flatten_rotations_preds = None
    if union3ds_preds:
        # self.visualization(union3ds_preds[0], img_metas, "/mnt/share_disk/zhangchen/data/visualization/v3.3_alldata/feature/union3d/small")
        # self.visualization(union3ds_preds[1], img_metas, "/mnt/share_disk/zhangchen/data/visualization/v3.3_alldata/feature/union3d/middle")
        # self.visualization(union3ds_preds[2], img_metas, "/mnt/share_disk/zhangchen/data/visualization/v3.3_alldata/feature/union3d/large")
        class_agnostic = True
        if not class_agnostic:
            flatten_union3ds_preds = [union3ds_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, union3ds_dims * num_classes) for union3ds_pred in union3ds_preds]
        else:
            flatten_union3ds_preds = [union3ds_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, union3ds_dims) for union3ds_pred in union3ds_preds]
        
        # torch.Size([1, 12096, 34])
        flatten_union3ds_preds = torch.cat(flatten_union3ds_preds, dim=1)

        # if not class_agnostic:
        #     assert len(flatten_union3ds_preds.shape) == 3
        #     b1, n2, c3 = flatten_union3ds_preds.shape
        #     flatten_union3ds_preds = flatten_union3ds_preds.view(b1, n2, -1, num_classes)

        #     _, classification_results = torch.max(flatten_cls_scores, 2)
        #     flatten_union3ds_preds = torch.gather(flatten_union3ds_preds, dim=3,
        #                                            index=classification_results.unsqueeze(-1).unsqueeze(-1).repeat(
        #                                                1, 1, c3 // self.num_classes, 1)).squeeze(-1)
        
        size_dims = 0
        flatten_dimensions_preds = flatten_union3ds_preds[..., 0:3]
        size_dims = 3
        flatten_positions_preds = flatten_union3ds_preds[..., 3:6]
        size_dims = 6
        # print(flatten_positions_preds[0][0], 3, 6)
        
        if use_depth_classifier==False:
            # 此处预测的位置对应自车坐标系的z,x,y,注意推理或测试时需转换成xyz的自车坐标系预测顺序
            flatten_camx_preds = flatten_positions_preds[...,0] #对应原始标注的自车坐标系的y
            flatten_camy_preds = flatten_positions_preds[...,1] #对应原始标注的自车坐标系的z
            flatten_camz_preds = flatten_positions_preds[...,2] #对应原始标注的自车坐标系的x
            # 在此处为flatten_positions_preds修正对应原始标注的车体坐标系的真实xyz标注顺序
            # print(flatten_camz_preds.shape)
            flatten_positions_preds = torch.stack((flatten_camz_preds, flatten_camx_preds,flatten_camy_preds), dim=2)
            # print(flatten_positions_preds)

        if size_dims<union3ds_dims:
            # yaw角预测，torch.Size([1, 12096, 1]), 
            flatten_rotations_preds = torch.atan2(flatten_union3ds_preds[..., 6:7] / 5.0 - 1.0, flatten_union3ds_preds[..., 7:8] / 5.0 - 1.0)
            size_dims = size_dims + 2
            # 增强yaw角预测     
            # if dir_cls_preds:
            #     # 方向预测，复用union3d的head输出的feats，预测器设置dir_bins=2，dir_bins为预测器输出channel维度
            #     flatten_dir_cls_preds = [
            #         dir_cls_pred[:,:self.dir_bins,:].permute(0, 2, 3, 1).reshape(num_imgs, -1, self.dir_bins) for dir_cls_pred in dir_cls_preds
            #     ]
            #     flatten_dir_cls_preds = torch.cat(flatten_dir_cls_preds, dim=1)
            #     flatten_dir_cls_preds = torch.max(flatten_dir_cls_preds, dim=-1)[1].unsqueeze(-1)
            #     #decode_yaw
            #     if flatten_rotations_preds.shape[0]>0:
            #         flatten_rotations_preds = self.decode_yaw(flatten_rotations_preds, flatten_dir_cls_preds, self.dir_offset)
            # print("@", flatten_rotations_preds.shape, flatten_rotations_preds)

            if size_dims<union3ds_dims:
                flatten_120_flag_preds = flatten_union3ds_preds[..., 8:10]
                # torch.Size([1, 12096])
                flatten_120_flag_preds = torch.max(flatten_120_flag_preds, dim=-1)[1]
                flatten_120_flag_preds_score = flatten_union3ds_preds[...,  8:10].sigmoid()
                # print(flatten_120_flag_preds_score)

            # onnx模型不输出
            # if self.extra_3d_points:
            #     start_dims += self.flag_120
            #     end_dims += self.corners_dims
            #     # torch.Size([1, 12096, 24])
            #     flatten_extra_3d_points_preds = flatten_union3ds_preds[..., start_dims:end_dims]

    if struct_preds:
        # 车灯状态分类，左转灯(unknown, on, off),右转灯(unknown, on, off),刹车灯(unknown, on, off) 3+3+3=9
        flatten_struct_preds = [struct_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 9) for struct_pred in struct_preds]
        flatten_struct_preds = torch.cat(flatten_struct_preds, dim=1).sigmoid()

    result_list = []
    for img_id in range(len(img_metas)):
        # torch.Size([12096, 3])
        cls_scores = flatten_cls_scores[img_id]
        # torch.Size([12096])
        score_factor = flatten_objectness[img_id]
        # torch.Size([12096, 4])
        bboxes = flatten_bboxes[img_id]

        # torch.Size([12096]), torch.Size([12096])
        max_scores, labels = torch.max(cls_scores, 1)
        # print(max_scores)
        valid_mask = score_factor * max_scores >= bbox_score_thr
        # if self.split_120:
        #     flag_120_mask = (flatten_120_flag_preds[img_id]==1)
        #     valid_mask = valid_mask & flag_120_mask

        # torch.Size([60, 4])
        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask] * score_factor[valid_mask]
        labels = labels[valid_mask]
        result = []
        if labels.numel() == 0:
            result.append(bboxes)
            result.append(scores)
            if attrs_preds:
                result.append(flatten_attrs_preds[img_id][valid_mask])
            # if self.pred_fake3d:
            #     result.append(flatten_fake3ds_flags[img_id][valid_mask])
            #     result.append(flatten_fake3ds_cords[img_id][valid_mask])
            if union3ds_preds:
                result.append(flatten_dimensions_preds[img_id][valid_mask])
                # if self.use_uvd:
                #     if self.use_uvd_position_consistency:
                #         result.append(flatten_positions_preds[img_id][valid_mask])
                #     # result.append(flatten_centerness_preds[img_id][valid_mask])
                #     # result.append(flatten_depths_preds[img_id][valid_mask])
                # else:
                result.append(flatten_positions_preds[img_id][valid_mask])

                if flatten_rotations_preds is  not None:
                    # yaw角预测，推理时候yaw = arctran(预测的坐标y, 预测的坐标x) + arctran((output1 / 5 - 1), 
                    #     (output2 / 5 - 1))，因为预测的角度值 = yaw - arctran(position_y, position_x)  
                    result.append(torch.atan2(flatten_positions_preds[img_id][valid_mask][...,1], 
                                                flatten_positions_preds[img_id][valid_mask][...,0]).unsqueeze(1)+ 
                                                flatten_rotations_preds[img_id][valid_mask])
            
                if flatten_120_flag_preds_score is not None:
                    result.append(flatten_120_flag_preds_score[img_id][valid_mask])
                
                # if self.extra_3d_points:
                #     result.append(flatten_extra_3d_points_preds[img_id][valid_mask])
            
            if struct_preds:
                result.append(flatten_struct_preds[img_id][valid_mask])

            result = torch.cat(result, dim=-1)
            result_list.append([result, labels])

        else:
            # Some type of nms would reweight the score, such as SoftNMS
            dets, keep = batched_nms(bboxes, scores, labels, bbox_nms)
            result.append(dets)#13*5
            if attrs_preds:
                result.append(flatten_attrs_preds[img_id][valid_mask][keep])#13*16
                
            # if self.pred_fake3d:
            #     result.append(flatten_fake3ds_flags[img_id][valid_mask][keep])#13*4
            #     result.append(flatten_fake3ds_cords[img_id][valid_mask][keep])#13*8
            
            if union3ds_preds:
                result.append(flatten_dimensions_preds[img_id][valid_mask][keep])#13*3
                # if self.use_uvd:
                #     if self.use_uvd_position_consistency:
                #         result.append(flatten_positions_preds[img_id][valid_mask][keep])
                #     # result.append(flatten_centerness_preds[img_id][valid_mask][keep])
                #     # result.append(flatten_depths_preds[img_id][valid_mask][keep])
                # else:
                result.append(flatten_positions_preds[img_id][valid_mask][keep])#13*3
                
                if flatten_rotations_preds is not None:
                    # yaw角预测，推理时候yaw = arctran(预测的坐标y, 预测的坐标x) + arctran((output1 / 5 - 1), 
                    #     (output2 / 5 - 1))，因为预测的角度值 = yaw - arctran(position_y, position_x)
                    result.append(torch.atan2(flatten_positions_preds[img_id][valid_mask][keep][...,1], 
                                                flatten_positions_preds[img_id][valid_mask][keep][...,0]).unsqueeze(1)+ 
                                                flatten_rotations_preds[img_id][valid_mask][keep])#13*1
                # camera coordinate
                # result.append(torch.atan2(flatten_positions_preds[img_id][valid_mask][keep][...,0], 
                # flatten_positions_preds[img_id][valid_mask][keep][...,2]).unsqueeze(1)+ flatten_rotations_preds[img_id][valid_mask][keep])
                # result.append(flatten_rotations_preds[img_id][valid_mask][keep])

                if flatten_120_flag_preds_score is not None:
                    result.append(flatten_120_flag_preds_score[img_id][valid_mask][keep])#13*2
                
                # if self.extra_3d_points:
                #     result.append(flatten_extra_3d_points_preds[img_id][valid_mask][keep])#13*24
            
            if struct_preds:
                result.append(flatten_struct_preds[img_id][valid_mask][keep])#13*9

            # torch-vehicle模型返回:75 = (bbox+score)5+(遮挡，方向等属性)16+12(左前,左后,右前,右后四个点是否可见4+四个点坐标8)
            #                      +(h,w,z)3+(xyz)3+(yaw)1+(是否120m内)2+(8个corners)24+(左转灯/右转灯/刹车灯)9
            # onnx-vehicle模型返回:39 = (bbox+score)5+(遮挡，方向等属性)16+(h,w,z)3+(xyz)3+(yaw)1+(是否120m内)2+(左转灯/右转灯/刹车灯)9
            # onnx-vru模型返回:24 = (bbox+score)5+(遮挡，方向等属性)10+(h,w,z)3+(xyz)3+(yaw)1+(是否120m内)2
            # onnx-static模型返回:11 = (bbox+score)5+(h,w,z)3+(xyz)3
            result = torch.cat(result, dim=-1)
            # labels表示检测的目标类别
            result_list.append([result, labels[keep]])
    
    # 把 decode_det_results() 预测的类别相同的实例预测list放到一个list里, 每个类别都会返回一个矩阵，即使预测为空也返回类似0*75的矩阵
    def bbox2result(bboxes, labels, num_classes, flag=False):

        shape = bboxes.shape[-1]
        if bboxes.shape[0] == 0:
            return [np.zeros((0, shape), dtype=np.float32) for i in range(num_classes)]
        else:
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
            return [bboxes[labels == i, :] for i in range(num_classes)]

    results = [bbox2result(det_bboxes, det_labels, num_classes)
               for det_bboxes, det_labels in result_list]

    # print("\n>>>>>>>>>task type = {}, results shape={}, ".format(task_type,  results[0][0].shape))
    return results




if __name__ == "__main__":
    prior_generator = MlvlPointGenerator([8, 16, 32], offset=0)
    
    mlv_priors = prior_generator.grid_priors([torch.Size([72, 128]), torch.Size([36, 64]), torch.Size([18, 32])], torch.device("cpu"), with_stride=True)
    
    print(torch.cat(mlv_priors).shape)
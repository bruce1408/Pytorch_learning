import os
import torch
import numpy as np

batch_size = 1
height = 430
width = 320
channels = 3
K = 80


def _gather_feat_nchw(feats, inds, feat_masks=None):
    """Given feats and indexes, returns the gathered feats.

    Args:
        feats (torch.Tensor): Features to be transposed and gathered
            with the shape of [B, 2, W, H].
        inds (torch.Tensor): Indexes with the shape of [B, N].
        feat_masks (torch.Tensor): Mask of the feats. Default: None.

    Returns:
        torch.Tensor: Gathered feats.
    """
    dim = feats.size(2)
    inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
    feats = feats.gather(1, inds)
    if feat_masks is not None:
        feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
        feats = feats[feat_masks]
        feats = feats.view(-1, dim)
    return feats


def _gather_feat_nhwc(feats, inds, feat_masks=None):
    """Given feats and indexes, returns the gathered feats.

    Args:
        feats (torch.Tensor): Features to be transposed and gathered
            with the shape of [B, H, W, C].
        inds (torch.Tensor): Indexes with the shape of [B, N].
        feat_masks (torch.Tensor): Mask of the feats. Default: None.

    Returns:
        torch.Tensor: Gathered feats.
    """
    batch_size, _, channels = feats.size()
    dim = channels

    # 将 inds 扩展到 [B, N, C] 维度
    inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)

    # 将 feats 展平为 [B, H*W, C]

    # Gather 操作
    gathered_feats = torch.gather(feats, 1, inds)

    # 如果提供了 feat_masks，应用掩码
    if feat_masks is not None:
        feat_masks = feat_masks.unsqueeze(2).expand_as(gathered_feats)
        gathered_feats = gathered_feats[feat_masks]
        gathered_feats = gathered_feats.view(-1, dim)

    return gathered_feats
    
    
def _topk_nhwc(scores, K=80):
    """Get indexes based on scores.

    Args:
        scores (torch.Tensor): scores with the shape of [B, H, W, C].
        K (int): Number to be kept. Defaults to 80.

    Returns:
        tuple[torch.Tensor]
            torch.Tensor: Selected scores with the shape of [B, K].
            torch.Tensor: Selected indexes with the shape of [B, K].
            torch.Tensor: Selected classes with the shape of [B, K].
            torch.Tensor: Selected y coord with the shape of [B, K].
            torch.Tensor: Selected x coord with the shape of [B, K].
    """
    batch, height, width, cat = scores.size()

    # 将 scores 展平为 [B, H*W*C]
    scores = scores.view(batch, -1)

    # 从所有类别中找到前 K 个最大值及其索引
    topk_scores, topk_inds = torch.topk(scores, K)

    # 计算类别索引、y 坐标和 x 坐标
    topk_clses = (topk_inds % cat).int()
    topk_inds = (topk_inds // cat)
    topk_ys = (topk_inds // width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def _topk_nchw(scores, K=80):
    """Get indexes based on scores.

    Args:
        scores (torch.Tensor): scores with the shape of [B, N, W, H].
        K (int): Number to be kept. Defaults to 80.

    Returns:
        tuple[torch.Tensor]
            torch.Tensor: Selected scores with the shape of [B, K].
            torch.Tensor: Selected indexes with the shape of [B, K].
            torch.Tensor: Selected classes with the shape of [B, K].
            torch.Tensor: Selected y coord with the shape of [B, K].
            torch.Tensor: Selected x coord with the shape of [B, K].
    """
    batch, cat, height, width = scores.size()

    # 将 scores 展平为 [B, N*W*H]
    scores = scores.view(batch, -1)

    # 从所有类别中找到前 K 个最大值及其索引
    topk_scores, topk_inds = torch.topk(scores, K)

    # 计算类别索引、y 坐标和 x 坐标
    topk_clses = (topk_inds / (height * width)).int()
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs



    

# 示例输入张量


scores_nchw = torch.tensor(np.arange(batch_size * channels * height * width)).reshape(batch_size, channels, height, width)
scores_nhwc = scores_nchw.permute(0, 2, 3, 1).contiguous()


# 调用 _topk 函数
topk_scores_chw, topk_inds_chw, topk_clses_chw, topk_ys_chw, topk_xs_chw = _topk_nchw(scores_nchw, K)
topk_scores_hwc, topk_inds_hwc, topk_clses_hwc, topk_ys_hwc, topk_xs_hwc = _topk_nhwc(scores_nhwc, K)
print(topk_scores_chw.shape, topk_inds_chw.shape, topk_clses_chw.shape, topk_ys_chw.shape, topk_xs_chw.shape)
print(topk_scores_hwc.shape, topk_inds_hwc.shape, topk_clses_hwc.shape, topk_ys_hwc.shape, topk_xs_hwc.shape)

print("scores", torch.equal(topk_scores_chw, topk_scores_hwc))
print("inds", torch.equal(topk_inds_chw, topk_inds_hwc))
print("clses", torch.equal(topk_clses_chw, topk_clses_hwc))
print("ys", torch.equal(topk_ys_chw, topk_ys_hwc))
print("xs", torch.equal(topk_xs_chw, topk_xs_hwc))


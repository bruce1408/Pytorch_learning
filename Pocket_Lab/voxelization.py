from typing import Tuple

import numpy as np


def voxelize(
        points: np.ndarray,
        voxel_size: np.ndarray,
        grid_range: np.ndarray,
        max_points_in_voxel: int,
        max_num_voxels: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts 3D point cloud to a sparse voxel grid
    :param points: (num_points, num_features), first 3 elements must be <x>, <y>, <z>
    :param voxel_size: (3,) - <width>, <length>, <height>
    :param grid_range: (6,) - <min_x>, <min_y>, <min_z>, <max_x>, <max_y>, <max_z>
    :param max_points_in_voxel:
    :param max_num_voxels:
    :param include_relative_position: boolean flag, if True, the output num_features will include relative
    position of the point within the voxel
    :return: tuple (
        voxels (num_voxels, max_points_in_voxels, num_features),
        coordinates (num_voxels, 3),
        num_points_per_voxel (num_voxels,)
    )
    """
    # 点云数据进行备份
    points_copy = points.copy()

    # 整个点云体素框的尺寸 [一共是432 x 496 x 1 个体素格子]，类似于图像尺寸3x3，一共是9个像素点
    grid_size = np.floor((grid_range[3:] - grid_range[:3]) / voxel_size).astype(np.int32)

    # 体素坐标对应的体素格子序号 [1 x 496 x 432]
    coor_to_voxelidx = np.full((grid_size[2], grid_size[1], grid_size[0]), -1, dtype=np.int32)

    # 体素框的数据维度=[体素格子数目 x 每个体素内最大点云数 x 点云的维度] = [16000 x 32 x 4]
    voxels = np.zeros((max_num_voxels, max_points_in_voxel, points.shape[-1]), dtype=points_copy.dtype)

    # 16000个体素格子的坐标 size = [16000, 3]
    coordinates = np.zeros((max_num_voxels, 3), dtype=np.int32)

    num_points_per_voxel = np.zeros(max_num_voxels, dtype=np.int32)

    # 每个点云在体素框里面的相对坐标位置
    points_coords = np.floor((points_copy[:, :3] - grid_range[:3]) / voxel_size).astype(np.int32)
    mask = ((points_coords >= 0) & (points_coords < grid_size)).all(1)

    # 点云坐标进行过滤
    points_coords = points_coords[mask, ::-1]

    # 点云数据进行过滤
    points_copy = points_copy[mask]
    assert points_copy.shape[0] == points_coords.shape[0]

    voxel_num = 0
    for i, coord in enumerate(points_coords):
        voxel_idx = coor_to_voxelidx[tuple(coord)]
        if voxel_idx == -1:
            voxel_idx = voxel_num
            if voxel_num > max_num_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[tuple(coord)] = voxel_idx
            coordinates[voxel_idx] = coord
        point_nums = num_points_per_voxel[voxel_idx]
        if point_nums < max_points_in_voxel:
            voxels[voxel_idx, point_nums] = points_copy[i]


            num_points_per_voxel[voxel_idx] += 1

    return voxels[:voxel_num], coordinates[:voxel_num], num_points_per_voxel[:voxel_num]


if __name__ == "__main__":
    voxel_size = np.array([0.16, 0.16, 4])
    point_cloud_range = np.array([0, -39.68, -3, 69.12, 39.68, 1])
    max_num_points = 32
    max_voxels = [16000, 40000]
    points = np.random.randn(30000, 4)

    vox, coord, _ = voxelize(points, voxel_size, point_cloud_range, max_num_points, max_voxels[0])
    print(vox.shape)
    print(coord.shape)

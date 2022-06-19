import numpy as np
import pyvista as pv
import h5py
import open3d as o3d

def calc_sorted_rms_distances(array: np.ndarray, point: np.ndarray):
    diff_array = array - point

    (diff_array_x, diff_array_y, diff_array_z) = (diff_array[..., 0], diff_array[..., 1], diff_array[..., 2])

    result = np.sqrt((diff_array_x**2 + diff_array_y**2 + diff_array_z**2))

    return np.size(np.where(np.abs(result) < 1.2)) - 1
    # return np.sqrt(np.mean(np.sort(result)[1:17] ** 2))

def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def main():
    sourcePC_path = '../source_point_clouds_preop_models/sk{}_face_d10000f.ply'.format(1)
    sourcePC_mesh = pv.read(sourcePC_path)

    faces = np.asarray(sourcePC_mesh.faces)
    points = np.asarray(sourcePC_mesh.points)

    # voxelize the point clouds here
    pcd0 = to_o3d_pcd(points)
    pcd0 = pcd0.voxel_down_sample(3.0)
    points = np.array(pcd0.points)

    dists = np.array([calc_sorted_rms_distances(points, point) for point in points])
    mean_dist_0 = np.mean(dists)

    folder = '../target_point_clouds/depthSensor'
    pcd_load = o3d.io.read_point_cloud('{}/p1_example_cleaned.ply'.format(folder))
    points = np.array(pcd_load.points)

    # voxelize the point clouds here
    pcd0 = to_o3d_pcd(points)
    pcd0 = pcd0.voxel_down_sample(3.0)
    pcd0 = pcd0.voxel_down_sample(2.0)
    points = np.array(pcd0.points)

    dists = np.array([calc_sorted_rms_distances(points, point) for point in points])
    mean_dist_1 = np.mean(dists)

    # filename = "../modelnet40_ply_hdf5_2048/ply_data_test0.h5"
    #
    # with h5py.File(filename, mode='r') as f:
    #     points = f['data'][1]
    #     pcd0 = to_o3d_pcd(points)
    #     pcd0 = pcd0.voxel_down_sample(0.06)
    #     points = np.array(pcd0.points)
    #     dists = np.array([calc_sorted_rms_distances(points, point) for point in points])
    #     mean_dist = np.mean(dists)

    print('Done!')


if __name__ == '__main__':
    main()

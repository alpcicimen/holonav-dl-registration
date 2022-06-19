import numpy as np
import pyvista as pv
import open3d as o3d

pc_number = 4
folder = '../target_point_clouds/depthSensor'
path = '{}/p{}.txt.npy'.format(folder, pc_number)

if __name__ == '__main__':

    targetPC = np.load(path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(targetPC)
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud('{}/p{}.txt.ply'.format(folder, pc_number), pcd)

    pcd_load = o3d.io.read_point_cloud('{}/p1_example_cleaned.ply'.format(folder))
    o3d.visualization.draw_geometries([pcd_load])

    pcd_load = pcd_load.voxel_down_sample(6.0)
    o3d.visualization.draw_geometries([pcd_load])

    print('done')

import sys
import numpy as np
import pyvista as pv
import h5py
import open3d as o3d

# filename = "../new_modelnet/new.h5"
filename = "../modelnet40_ply_hdf5_2048/ply_data_test0.h5"

if __name__ == '__main__':
    f = h5py.File(filename, mode='r')

    points = f['data'][:]
    normals = f['normal'][:]
    labels = f['label'][:].flatten().astype(np.int64)

    # pcd = list()
    #
    # for i in range(0, 3):
    #     pcd.append(o3d.geometry.PointCloud())
    #     pcd[i].points = o3d.utility.Vector3dVector(points[i])
    #     pcd[i].normals = o3d.utility.Vector3dVector(normals[i])

    # i: int = 0

    for model in points:
        x_max = np.max(model[:, 0])
        y_max = np.max(model[:, 1])
        z_max = np.max(model[:, 2])

        x_min = np.min(model[:, 0])
        y_min = np.min(model[:, 1])
        z_min = np.min(model[:, 2])

        print("Max X: %f, Max Y: %f, Max Z: %f" % (x_max, y_max, z_max))
        print("Min X: %f, Min Y: %f, Min Z: %f" % (x_min, y_min, z_min))

        volume = np.abs(x_max - x_min) * np.abs(y_max - y_min) * np.abs(z_max - z_min)

        c_point = np.array([(x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2])

        # pcd[i].scale(1/(100), c_point)
        #
        # i = i + 1

        print("Volume: %f" % volume)

    # for pc in pcd:
    #     x_max = np.max(np.asarray(pc.points)[:, 0])
    #     y_max = np.max(np.asarray(pc.points)[:, 1])
    #     z_max = np.max(np.asarray(pc.points)[:, 2])
    #
    #     x_min = np.min(np.asarray(pc.points)[:, 0])
    #     y_min = np.min(np.asarray(pc.points)[:, 1])
    #     z_min = np.min(np.asarray(pc.points)[:, 2])
    #
    #     print("Max X: %f, Max Y: %f, Max Z: %f" % (x_max, y_max, z_max))
    #     print("Min X: %f, Min Y: %f, Min Z: %f" % (x_min, y_min, z_min))
    #
    #     volume = np.abs(x_max - x_min) * np.abs(y_max - y_min) * np.abs(z_max - z_min)
    #
    #     print("Volume: %f" % volume)

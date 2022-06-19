import random

import numpy as np
import pyvista as pv
import h5py
import open3d as o3d

skull_number = 1
pc_number = 4
target_pc = 'optical_reg'   # options: 'optical_raw', 'optical_reg', and 'pre-op'
show_normals = True

# filename_h5 = "../modelnet40_ply_hdf5_2048/ply_data_test0.h5"
filename_h5 = "../new_modelnet/new.h5"

if __name__ == '__main__':

    f = h5py.File(filename_h5, mode='a')

    data = np.array([])
    label = np.array([])
    normal = np.array([])

    for i in range(1, 4):

        sourcePC_path = '../source_point_clouds_preop_models/sk{}_face_d10000f.ply'.format(i)

        sourcePC: o3d.geometry.PointCloud = o3d.io.read_point_cloud(sourcePC_path)
        sourcePC_mesh = pv.read(sourcePC_path)

        # data = np.append(data, np.array([np.asarray(sourcePC.points)]), axis=0)\
        #     if len(data > 0) else np.array([np.asarray(sourcePC.points)])
        # label = np.append(label, 22)
        # normal = np.append(normal, np.array([np.asarray(sourcePC_mesh.point_normals)]), axis=0)\
        #     if len(normal>0) else np.array([np.asarray(sourcePC_mesh.point_normals)])

        c_point = np.array([0.0, 0.0, 0.0])
        sourcePC.scale(1 / (100), c_point)

        pc_points = np.asarray(sourcePC.points)
        pc_normals = np.asarray(sourcePC_mesh.point_normals)

        while len(pc_points) > 1024:
            r_index = random.randint(0, len(pc_points) - 1)
            pc_points = np.delete(pc_points, r_index, 0)
            pc_normals = np.delete(pc_normals, r_index, 0)

        # data = data + [np.asarray(sourcePC.points)[:5080]] \
        #     if len(data) > 0 else [np.asarray(sourcePC.points)[:5080]]
        # label = np.append(label, 22)
        # normal = normal + [np.asarray(sourcePC_mesh.point_normals)[:5080]] \
        #     if len(normal) > 0 else [np.asarray(sourcePC_mesh.point_normals)[:5080]]

        data = data + [pc_points] \
            if len(data) > 0 else [pc_points]
        label = np.append(label, 22)
        normal = normal + [pc_normals] \
            if len(normal) > 0 else [pc_normals]



    data = np.array(data)
    normal = np.array(normal)

    f.create_dataset(name='data', data=data, dtype=np.float32)
    f.create_dataset(name='label', data=label, dtype=np.int64)
    f.create_dataset(name='normal', data=normal, dtype=np.float32)

def readh5():
    f = h5py.File(filename_h5, mode='a')
import numpy as np
import pyvista as pv
import h5py

filename = "../modelnet40_ply_hdf5_2048/ply_data_test0.h5"

if __name__ == '__main__':

    with h5py.File(filename, mode='r') as f:

        points = f['data'][:]
        normals = f['normal'][:]
        labels = f['label'][:].flatten().astype(np.int64)

        for i, _ in enumerate(points):
            pc_points = points[i]
            pc_normals = normals[i]
            pc_label = labels[i]

            print(i)

            pc = pv.PolyData(pc_points)
            pc['Normals'] = pc_normals

            pc.plot_normals(mag=0.05)

        print('done')
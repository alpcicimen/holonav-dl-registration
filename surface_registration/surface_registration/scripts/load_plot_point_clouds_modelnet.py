import h5py
import numpy as np
import pyvista as pv

skull_number = 2
pc_number = 3
target_pc = 'optical_reg'   # options: 'optical_raw', 'optical_reg', and 'pre-op'
show_normals = True

filename = "../modelnet40_ply_hdf5_2048/ply_data_test0.h5"

sourcePC_path = '../source_point_clouds_preop_models/sk{}_face_d10000f.ply'.format(skull_number)
if target_pc == 'optical_raw':
    targetPC_path = '../target_point_clouds/Optical/points1/sk{}/pc{}.txt'.format(skull_number, pc_number)
elif target_pc == 'optical_reg':
    targetPC_path = '../target_point_clouds/Optical/points1/sk{}/reg_pc{}.txt'.format(skull_number, pc_number)
else:
    targetPC_path = '../target_point_clouds/PreOp/sk{}/pc{}.txt'.format(skull_number, pc_number)

if __name__ == '__main__':
    sourcePC_mesh = pv.read(sourcePC_path)

    targetPC = np.loadtxt(targetPC_path)

    targetPC_points = pv.PolyData(targetPC)
    targetPC_mesh = targetPC_points.delaunay_2d()

    if show_normals:
        targetPC_mesh.plot_normals(mag=10, flip=True)

    with h5py.File(filename, mode='r') as f:

        points = f['data'][:]
        normals = f['normal'][:]
        labels = f['label'][:].flatten().astype(np.int64)

        pc_points = points[0]
        pc_normals = normals[0]
        pc_label = labels[0]

        # pc = pv.PolyData(pc_points)
        # pc['Normals'] = pc_normals
        #
        # pc.plot_normals(mag=0.05)

        p = pv.Plotter()
        p.add_mesh(sourcePC_mesh, smooth_shading=True, opacity=0.3)
        p.add_points(targetPC_points, smooth_shading=True, color='blue')
        p.add_points(pc_points, smooth_shading=True, color='red')
        p.show()

    print('done')

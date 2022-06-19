import h5py
import numpy as np
import pyvista as pv
import open3d as o3d

skull_number = 1
pc_number = 4
target_pc = 'optical_reg'   # options: 'optical_raw', 'optical_reg', and 'pre-op'
show_normals = True

sourcePC_path = '../source_point_clouds_preop_models/sk{}_face_d10000f.ply'.format(skull_number)


if __name__ == '__main__':

    sourcePC = o3d.io.read_point_cloud(sourcePC_path)

    sourcePC_mesh = pv.read(sourcePC_path)

    pc = pv.PolyData(np.asarray(sourcePC.points))

    pc['Normals'] = np.asarray(sourcePC_mesh.point_normals)

    pc.plot_normals(mag=2)

    print('done')
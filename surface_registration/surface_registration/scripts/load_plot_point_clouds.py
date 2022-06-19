import numpy as np
import pyvista as pv

skull_number = 3
pc_number = 3
target_pc = 'optical_reg'   # options: 'optical_raw', 'optical_reg', and 'pre-op'
show_normals = True

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


    p = pv.Plotter()
    p.add_mesh(sourcePC_mesh, smooth_shading=True, opacity=0.3)
    p.add_points(targetPC_points, smooth_shading=True, color='blue')
    p.show()

    print('done')

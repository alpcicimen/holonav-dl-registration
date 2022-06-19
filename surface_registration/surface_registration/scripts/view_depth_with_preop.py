import numpy as np
import pyvista as pv
import open3d as o3d

folder = '../target_point_clouds/depthSensor'
pcd_load = o3d.io.read_point_cloud('{}/p1_example_cleaned.ply'.format(folder))
skull_number = 1
sourcePC_path = '../source_point_clouds_preop_models/sk{}_face_d10000f.ply'.format(skull_number)
show_normals = True


def main():
    # o3d.visualization.draw_geometries([pcd_load])
    points = np.array(pcd_load.points)
    sourcePC_mesh = pv.read(sourcePC_path)
    targetPC_points = pv.PolyData(points)
    targetPC_mesh = targetPC_points.delaunay_2d()

    if show_normals:
        targetPC_mesh.plot_normals(mag=10, flip=True)

    p = pv.Plotter()
    p.add_mesh(sourcePC_mesh, smooth_shading=True, opacity=0.3)
    p.add_points(targetPC_points, smooth_shading=True, color='blue')
    p.show()

    print('done')


if __name__ == '__main__':
    main()

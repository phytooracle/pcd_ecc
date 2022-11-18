import numpy as np
import euchar.utils
from euchar.curve import image_2D, image_3D, filtration
from euchar.filtrations import alpha_filtration_2D, alpha_filtration_3D, inverse_density_filtration
from euchar.display import piecewise_constant_curve
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
plt.rcParams.update({"font.size": 16})
from seaborn import distplot,displot,histplot
import open3d as o3d

def visualize(obj):
    print(type(obj))
    o3d.visualization.draw_geometries([obj])

def voxelization(filename):
    N = 2000

    pcd = o3d.io.read_point_cloud(filename)
    
    # Normalize point cloud points
    xyz = np.array(pcd.points)
    points = xyz - np.expand_dims(np.min(xyz, axis=0), 0)

    # create new point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # voxelization
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.001)

    return pcd,voxel_grid

def voxels_to_img3d(voxel):
    voxels = voxel.get_voxels()
    indices = np.stack(list(vx.grid_index for vx in voxels))
    colors = np.stack(list(vx.color for vx in voxels))

    min_, max_ = 1000000, -100000
    for index in indices:
        min_ = min(min_, index[0], index[1], index[2])
        max_ = max(max_, index[0], index[1], index[2])
    
    max_ += 1

    img3d = np.full((max_, max_, max_), 255)
    for i in range(len(indices)):
        x, y, z = indices[i]
        img3d[x,y,z] = 0

    return img3d

def euler_char_curves(points, img3d, testcase):
    #vector_3D_changes = euchar.utils.vector_all_euler_changes_in_3D_images()
    #np.save("vector_3D_changes.npy", vector_3D_changes)
    #vector_3D_changes = np.load("vector_3D_changes.npy")
    #ecc_3D = image_3D(img3d, vector_3D_changes)

    #plt.plot(np.arange(256), ecc_3D, color="blue")
    #plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(14,4))
    plt.subplots_adjust(wspace=0.3)

    simplices_3D, alpha_3D = alpha_filtration_3D(points)
    
    histplot(alpha_3D, ax=ax[0])
    ax[0].set(title="Distribution 3D miniball radiuses - "+testcase)

    bins_3D = np.linspace(0.0, 1, num=200)
    filt_3D = filtration(simplices_3D, alpha_3D, bins_3D)
    
    ax[1].plot(bins_3D, filt_3D, color="royalblue")
    ax[1].set(title="Euler char curve - 3D points - "+testcase, xlim=[-0.02, 1.02], ylim=[-50, 150]);
    
    plt.show()

def run(testcase, index):
    pcd,voxel = voxelization(testcase)
    visualize(pcd)
    visualize(voxel)
    img3d = voxels_to_img3d(voxel)
    points = np.asarray(pcd.points)
    euler_char_curves(points, img3d, str(index+1))

def main():
    testcases = ["test_no_soil_full_plant1.ply", "test_no_soil_full_plant2.ply"]

    run(testcases[1], 1)
    
    return

if __name__=='__main__':
    main()

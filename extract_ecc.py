#!/usr/bin/python3
import sys
import os
import argparse as arg
import numpy as np
import euchar.utils
from euchar.curve import image_2D, image_3D, filtration
from euchar.filtrations import alpha_filtration_2D, alpha_filtration_3D, inverse_density_filtration
from euchar.display import piecewise_constant_curve
import matplotlib.pyplot as plt
from seaborn import distplot,displot,histplot
import open3d as o3d

def visualize(obj):
    '''visualize open3d object'''

    o3d.visualization.draw_geometries([obj])

def voxelization(filename, voxel_size):
    '''converts a pcd file and returns PCD and VoxelGrid object'''

    pcd = o3d.io.read_point_cloud(filename)
    
    # Normalize point cloud points
    xyz = np.array(pcd.points)
    points = xyz - np.expand_dims(np.min(xyz, axis=0), 0)

    # create new point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    N = len(pcd.points)

    # voxelization
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

    return pcd,voxel_grid

def voxels_to_img3d(voxel):
    '''return 3d array of pixel values from voxel object'''

    voxels = voxel.get_voxels()
    indices = np.stack(list(vx.grid_index for vx in voxels))
    colors = np.stack(list(vx.color for vx in voxels))

    min_, max_ = 1000000, -100000
    for index in indices:
        min_ = min(min_, index[0], index[1], index[2])
        max_ = max(max_, index[0], index[1], index[2])
    
    max_ += 1

    img3d = np.ones((max_, max_, max_))
    for i in range(len(indices)):
        x, y, z = indices[i]
        img3d[x,y,z] = 0

    return img3d

def euler_char_curves(points, img3d):
    '''extract ecc from set of points and array of pixel values of 3d image'''

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
    ax[0].set(title="Distribution 3D miniball radiuses")

    bins_3D = np.linspace(0.0, 1, num=200)
    filt_3D = filtration(simplices_3D, alpha_3D, bins_3D)
    
    ax[1].plot(bins_3D, filt_3D, color="royalblue")
    ax[1].set(title="Euler char curve - 3D points", xlim=[-0.02, 1.02], ylim=[-50, 150]);
    
    plt.show()

def run(file, p, v, voxel_size):
    '''visualize 3d image file as pcd and voxels, extract and display ecc plots'''

    pcd,voxel = voxelization(file, voxel_size)
    if p:
        visualize(pcd)
    if v:
        visualize(voxel)
    img3d = voxels_to_img3d(voxel)
    points = np.asarray(pcd.points)
    euler_char_curves(points, img3d)

def error_handling_args(args):
    try:
        file = open(args.filename)
        file.close()
    except IOError:
        print("Could not open/read file:", args.filename)
        sys.exit()

def main():
    def_vs = 0.001
    parser = arg.ArgumentParser()

    # arguments/flags to run
    parser.add_argument("filename", help="Input PCD file")
    parser.add_argument("-p", "--pcd", action="store_true", help="Visualize as PCD image")
    parser.add_argument("-v", "--voxel", action="store_true", help="Visualize as Voxel image")
    parser.add_argument("-vs", "--voxel-size", help=f"Set voxel size (default {def_vs:g})", default=def_vs, type=float)

    args = parser.parse_args()

    # error handling on argument values
    error_handling_args(args)

    # run with arguments/flags
    run(args.filename, args.pcd, args.voxel, args.voxel_size)

    return

if __name__=='__main__':
    main()

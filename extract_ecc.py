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
import pandas as pd

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

def euler_char_curves(points, img3d, save, output, plant_name, visualize_ecc):
    '''extract ecc from set of points and array of pixel values of 3d image'''

    outpath = os.path.join(os.getcwd(), output, "figures", plant_name)

    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    simplices_3D, alpha_3D = alpha_filtration_3D(points)
    bins_3D = np.linspace(0.0, 1, num=200)
    filt_3D = filtration(simplices_3D, alpha_3D, bins_3D)

    fig, ax = plt.subplots(1, 2, figsize=(14,4))
    plt.subplots_adjust(wspace=0.3)

    histplot(alpha_3D, ax=ax[0])
    ax[0].set(title="Distribution 3D miniball radiuses")

    ax[1].plot(bins_3D, filt_3D, color="royalblue")
    ax[1].set(title="Euler char curve - 3D points", xlim=[-0.02, 1.02], ylim=[-50, 150]);

    plt.savefig(os.path.join(outpath, '_'.join([plant_name, "ecc_figure.png"])), dpi=900, bbox_inches='tight', facecolor='white', edgecolor='white')

    if visualize_ecc:
        plt.show()
        
    if save:
        outpath = os.path.join(os.getcwd(), output, "dataframes", plant_name)

        if not os.path.isdir(outpath):
            os.makedirs(outpath)
        
        df = pd.DataFrame({'bin': bins_3D, 'filter': filt_3D})
        
        df.to_csv(os.path.join(outpath, '_'.join([plant_name, "ecc.csv"])))

        save_array(array=simplices_3D, output=output, plant_name=plant_name, tag="simp_3D")
        save_array(array=alpha_3D, output=output, plant_name=plant_name, tag="alpha_3D")
        save_array(array=bins_3D, output=output, plant_name=plant_name, tag="bins_3D")
        save_array(array=filt_3D, output=output, plant_name=plant_name, tag="filt_3D")

def save_array(array, output, plant_name, tag):

    outpath = os.path.join(os.getcwd(), output, "arrays", plant_name)

    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    np.save(os.path.join(outpath, '_'.join([plant_name, tag])), array)

def run(file, p, v, voxel_size, save, output, plant_name, visualize_ecc):
    '''visualize 3d image file as pcd and voxels, extract and display ecc plots'''

    pcd,voxel = voxelization(file, voxel_size)
    if p:
        visualize(pcd)
    if v:
        visualize(voxel)
    img3d = voxels_to_img3d(voxel)
    points = np.asarray(pcd.points)
    euler_char_curves(points=points, img3d=img3d, save=save, output=output, plant_name=plant_name, visualize_ecc=visualize_ecc)

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
    parser.add_argument("-p", "--pcd", action="store_true", help="Visualize as PCD image. Default: False")
    parser.add_argument("-v", "--voxel", action="store_true", help="Visualize as Voxel image. Default: False")
    parser.add_argument("-vis", "--visualize_ecc", action="store_true", help="Visualize ECC curve. Default: False")
    parser.add_argument("-s", "--save", action="store_false", help="Store numpy of ECC data. Default: True")
    parser.add_argument("-vs", "--voxel-size", help=f"Set voxel size (default {def_vs:g})", default=def_vs, type=float)
    parser.add_argument("-n", "--name", required=True, help="The name of the input plant.")
    parser.add_argument("-o", "--output", default="euler_characteristic_curves", help="The name of the output directory.")

    args = parser.parse_args()

    # error handling on argument values
    error_handling_args(args)

    # run with arguments/flags
    run(file=args.filename, p=args.pcd, v=args.voxel, voxel_size=args.voxel_size, save=args.save, output=args.output, plant_name=args.name, visualize_ecc=args.visualize_ecc)

    return

if __name__=='__main__':
    main()

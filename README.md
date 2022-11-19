# ECC extraction tool
Extract Euler Characteristic Curves (ECC) from 3D point cloud data (PCD).

usage: extract_ecc.py [-h] [-p] [-v] [-vs VOXEL_SIZE] filename

positional arguments:
  filename              Input PCD file

options:
  -h, --help            show this help message and exit
  -p, --pcd             Visualize as PCD image
  -v, --voxel           Visualize as Voxel image
  -vs VOXEL_SIZE, --voxel-size VOXEL_SIZE
                        Set voxel size (default 0.001)

# -*- coding: utf-8 -*-


import os
import sys

import numpy as np
import trimesh

import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes, generate_binary_structure
import vtk
from natsort import natsorted
import re
import json
import shutil


def get_affine_from_vtk(nifti_path):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nifti_path)
    reader.Update()
    spacing = reader.GetOutput().GetSpacing()
    numpy_matrix = np.zeros((4, 4))
    numpy_matrix[0][0] = spacing[0]
    numpy_matrix[1][1] = spacing[1]
    numpy_matrix[2][2] = spacing[2]
    numpy_matrix[3][3] = 1.0
    return numpy_matrix


def get_affine_from_itk(im_sitk):
    spacing = im_sitk.GetSpacing()
    direction = im_sitk.GetDirection()
    origin = im_sitk.GetOrigin()
    affine = np.eye(4)
    affine[0, :3] = np.asarray(direction[:3]) * spacing[0]
    affine[1, :3] = np.asarray(direction[3:6]) * spacing[1]
    affine[2, :3] = np.asarray(direction[6:9]) * spacing[2]
    affine[0, 3] = origin[0]
    affine[1, 3] = origin[1]
    affine[2, 3] = origin[2]
    return affine


def save_itk_new(image, filename, origin, spacing, direction):
    if type(origin) != tuple:
        if type(origin) == list:
            origin = tuple(reversed(origin))
        else:
            origin = tuple(reversed(origin.tolist()))
    if type(spacing) != tuple:
        if type(spacing) == list:
            spacing = tuple(reversed(spacing))
        else:
            spacing = tuple(reversed(spacing.tolist()))
    if type(direction) != tuple:
        if type(direction) == list:
            direction = tuple(reversed(direction))
        else:
            direction = tuple(reversed(direction.tolist()))
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    itkimage.SetDirection(direction)
    sitk.WriteImage(itkimage, filename, True)


def load_itk_image_new(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = list(reversed(itkimage.GetOrigin()))
    numpySpacing = list(reversed(itkimage.GetSpacing()))
    numpyDirection = list(reversed(itkimage.GetDirection()))
    return numpyImage, numpyOrigin, numpySpacing, numpyDirection


def implicit_skeleton(in_mesh_path, transinfo_path, image_in_nii_path, image_out_nii_path, save_flag=False):
    mesh = trimesh.load(in_mesh_path)
    vertices = mesh.vertices
    with open(transinfo_path, 'r') as f:
        transinfo = json.load(f)
    scale = transinfo['s']
    R = np.array(transinfo['R'])
    t = np.array(transinfo['t'])
    Rt = np.linalg.inv(R)
    vertices = np.dot(((vertices - t) / scale), Rt)
    mesh_subdivided_vertices, mesh_subdivided_faces = trimesh.remesh.subdivide_loop(vertices, mesh.faces, iterations=2)
    mesh = trimesh.Trimesh(vertices=mesh_subdivided_vertices, faces=mesh_subdivided_faces)
    vertices = mesh.vertices
    im, origin, spacing, direction = load_itk_image_new(image_in_nii_path)
    affine = get_affine_from_vtk(image_in_nii_path)
    affine_t = np.linalg.inv(affine)
    vertices = np.transpose(vertices)
    ones = np.ones((1, vertices.shape[1]))
    vertices = np.concatenate([vertices, ones])
    vertices = np.matmul(affine_t, vertices)
    vertices = vertices[:3, :]
    vertices = np.transpose(vertices)
    vertices = np.around(vertices).astype(np.int16)
    vertices = vertices[:, ::-1]
    output_volume = np.zeros(im.shape, dtype=np.uint8)
    output_volume[vertices[:, 0], vertices[:, 1], vertices[:, 2]] = 1
    output_volume = output_volume.astype(np.uint8)
    output_volume_filled = binary_fill_holes(output_volume).astype(np.uint8)
    if save_flag:
        save_itk_new(output_volume_filled, image_out_nii_path, origin, spacing, direction)


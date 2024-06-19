# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import trimesh
import SimpleITK as sitk
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes, generate_binary_structure
import vtk
from natsort import natsorted
import re
import json
import shutil


def discreate_marching_cubes(nii_file_path, save_file_path, save_suffix='stl', use_binary=False):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nii_file_path)
    reader.Update()
    iso = vtk.vtkDiscreteMarchingCubes()
    iso.SetInputConnection(reader.GetOutputPort())
    iso.ComputeNormalsOn()
    iso.ComputeGradientsOn()
    iso.SetValue(0, 1)
    iso.Update()
    if save_suffix == 'stl':
        writer = vtk.vtkSTLWriter()
        writer.SetInputConnection(iso.GetOutputPort())
        if use_binary:
            writer.SetFileTypeToBinary()
        writer.SetFileName(save_file_path)
        writer.Write()
    elif save_suffix == 'obj':
        writer = vtk.vtkOBJWriter()
        writer.SetFileName(save_file_path)
        writer.SetInputData(iso.GetOutput())
        writer.Write()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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


def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)
    return trimesh.Trimesh(vertices=vertices, faces=None)


def map_normalized_mesh_back(original_mesh_path, normalized_mesh_path, save_path=None, unitsphere=True,
                             halfsphere=False):
    if unitsphere and not halfsphere:
        original_mesh: trimesh.Trimesh = trimesh.load(original_mesh_path)
        original_mesh_center = original_mesh.bounding_box.centroid
        original_mesh_distances = np.linalg.norm((original_mesh.vertices - original_mesh.bounding_box.centroid), axis=1)
        normalized_mesh: trimesh.Trimesh = trimesh.load(normalized_mesh_path)
        normalized_mesh_vertices = normalized_mesh.vertices
        normalized_mesh_vertices *= np.max(original_mesh_distances)
        normalized_mesh_vertices += original_mesh_center
        back_mesh = trimesh.Trimesh(vertices=normalized_mesh_vertices, faces=normalized_mesh.faces)
    else:
        raise ValueError('the normalization method is not defined!')
    if save_path:
        back_mesh.export(save_path)
    return back_mesh


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


def locate_pairwise_distance(nii_path, mesh_path):
    affine = get_affine_from_vtk(nii_path)
    mesh: trimesh.Trimesh = trimesh.load(mesh_path)
    connected_components = mesh.split(only_watertight=False)
    size_threshold = 10
    large_components = [c for c in connected_components if c.faces.shape[0] > size_threshold]
    large_components.sort(key=lambda x: x.faces.shape[0], reverse=True)
    sub_mesh_1 = large_components[0]
    sub_mesh_2 = large_components[1]
    sub_mesh_1_vertices = sub_mesh_1.vertices
    sub_mesh_2_vertices = sub_mesh_2.vertices
    sub_mesh_1_vertices_kdtree = cKDTree(sub_mesh_1_vertices)
    distances, indices = sub_mesh_1_vertices_kdtree.query(sub_mesh_2_vertices)
    min_dist_index = np.argmin(distances)
    closest_point_in_set_1 = sub_mesh_1_vertices[indices[min_dist_index]]
    closest_point_in_set_2 = sub_mesh_2_vertices[min_dist_index]
    radius_distance = distances[min_dist_index] / 2 * 3
    breakage_center_point_worcor = (closest_point_in_set_1 + closest_point_in_set_2) / 2
    breakage_center_point_worcor = (closest_point_in_set_1 + closest_point_in_set_2) / 2
    return breakage_center_point_worcor, radius_distance


def denoise_and_completion_with_normal_correction(nii_path, corrupted_data_path, completion_points_path,
                                                  refined_save_path, breakage_center_point_worcor, radius_distance,
                                                  lamda_r=1.5, k_nearest_normal=10):
    def remove_rows(Nx3_array, Mx3_array):
        mask = np.isin(Nx3_array, Mx3_array).all(1)
        result = Nx3_array[~mask]
        return result

    def remove_matching_rows(Nx6_array, Mx3_array):
        Nx3_part = Nx6_array[:, :3]
        mask = np.isin(Nx3_part, Mx3_array).all(1)
        result = Nx6_array[~mask]
        return result

    def matching_rows(Nx6_array, Mx3_array):
        Nx3_part = Nx6_array[:, :3]
        mask = np.isin(Nx3_part, Mx3_array, assume_unique=True).all(1)
        result = Nx6_array[mask]
        return result

    nii_image = sitk.ReadImage(nii_path)
    affine = get_affine_from_itk(nii_image)
    corrupted_data = np.loadtxt(corrupted_data_path)
    corrupted_data_points = corrupted_data[:, 0:3]
    corrupted_data_normals = corrupted_data[:, 3:]
    corrupted_data_points_kdtree = KDTree(data=corrupted_data_points)
    noised_points_indices = corrupted_data_points_kdtree.query_ball_point(x=breakage_center_point_worcor,
                                                                          r=radius_distance)
    noised_points_array = corrupted_data_points_kdtree.data[noised_points_indices]

    corrupted_data_points_denoise = remove_rows(corrupted_data_points, noised_points_array)
    corrupted_data_denoise = remove_matching_rows(corrupted_data, noised_points_array)
    completion_points: trimesh.Trimesh = trimesh.load(completion_points_path)
    completion_points_array = completion_points.vertices
    completion_points_array_kdtree = KDTree(data=completion_points_array)
    completed_points_indices = completion_points_array_kdtree.query_ball_point(x=breakage_center_point_worcor,
                                                                               r=lamda_r * radius_distance)
    completed_points_indices_array = completion_points_array_kdtree.data[completed_points_indices]

    corrupted_data_denoise_kdtree = KDTree(corrupted_data_points_denoise)
    corrected_normal_list = []
    for idx in range(0, completed_points_indices_array.shape[0]):
        dd, ii = corrupted_data_denoise_kdtree.query(completed_points_indices_array[idx], k=k_nearest_normal)
        k_nearest_data = matching_rows(corrupted_data_denoise, corrupted_data_denoise_kdtree.data[ii])
        mean_normal = np.mean(k_nearest_data, axis=0)[3:]
        corrected_normal_list.append(mean_normal)
    corrected_normal_array = np.asarray(corrected_normal_list)
    completed_points_indices_array_normal = np.concatenate((completed_points_indices_array, corrected_normal_array),
                                                           axis=1)
    final_points_array = np.concatenate((corrupted_data_points_denoise, completed_points_indices_array), axis=0)
    final_points = np.concatenate((corrupted_data_denoise, completed_points_indices_array_normal), axis=0)
    np.savetxt(refined_save_path, final_points, fmt='%.06f')

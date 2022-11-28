import glob

from PIL import Image
import os
import numpy as np
import torch
import torch.nn.functional as F
from .dcm_utils import read_dcm

def count_coronal(patient_id, xray_image_dir, threshold=0.1):

    patient_dir = os.path.join(xray_image_dir, patient_id)

    coronal = Image.open(os.path.join(patient_dir, "coronal.jpeg"))
    arr = np.asarray(coronal)

    num_columns = arr.shape[1]

    column_counts = np.zeros(num_columns)
    for i in range(num_columns):
        column = arr[:, i]
        idx = np.greater(column, threshold).nonzero()[0]
        column_counts[i] = len(idx)

    return column_counts, arr


# def find_coronal_center(patient_id, xray_image_dir, threshold=0.1, cutline=0.8):
#     """
#     find center x value from 3d coronal image
#     예를들어 8693 같은 이미지에서는 성적이 좋지 않았다.
#     :param patient_id:
#     :param xray_image_dir:
#     :param threshold:
#     :param cutline:
#     :return:
#     """
#     column_counts, arr = count_coronal(patient_id, xray_image_dir, threshold)
#
#     column_counts[column_counts < arr.shape[0] * cutline] = 0
#     nonzero_idx = column_counts.nonzero()[0]
#     center = (nonzero_idx[0] + nonzero_idx[-1]) // 2
#     return center

def find_coronal_center(patient_id, xray_image_dir, threshold=0.1, pick=100):
    """
    수직으로 통계내서 수치가 가장 많은 백개의 평균을 취한다. 
    :param patient_id:
    :param xray_image_dir:
    :param threshold:
    :param pick:
    :return:
    """
    coronal_img = Image.open(os.path.join(xray_image_dir, patient_id, "coronal.jpeg"))
    coronal = np.asarray(coronal_img) / 255.
    coronal[coronal < threshold] = 0
    coronal[coronal > 0] = 1
    count_columns = np.mean(coronal, axis=0)
    center_column = np.mean(np.argsort(count_columns)[-pick:])
    return center_column, coronal_img

def find_sagittal_top(patient_id, xray_images_dir, train_images_dir, aspect, pixel_spacing, z_flip, slice_list, center_column=None, threshold=0.1):

    if center_column is None:
        center_column, _ = find_coronal_center(patient_id, xray_images_dir)

    sagittal = Image.open(os.path.join(xray_images_dir, patient_id, "sagittal.jpeg"))
    sagittal = np.asarray(sagittal) / 255.
    sagittal_tensor = torch.tensor(sagittal)
    sagittal_tensor[sagittal_tensor < threshold] = 0
    sagittal_tensor = sagittal_tensor.reshape(1, 1, *sagittal_tensor.shape)
    pool_out = F.avg_pool2d(sagittal_tensor, kernel_size=(10, 10), stride=1, padding=(5, 5)).squeeze()

    nonzero_indices = pool_out.nonzero()
    max_columns = np.zeros(pool_out.shape[0])
    for i in range(nonzero_indices.shape[0]):
        row, column = nonzero_indices[i]
        max_columns[row] = column

    # min slice 가 있는 위치에서 두개골과 갈비뼈의 영향을 받지 않는다. 
    min_slice = max_columns.argmin()
    min_slice = int(np.round(min_slice / aspect))
    min_bottom = int(max_columns.min())

    if z_flip == True:
        num_instance = slice_list[-min_slice-1]
    else:
        num_instance = slice_list[min_slice]
    slice_img = read_dcm(os.path.join(train_images_dir, patient_id), num_instance)

    # 이걸 더 작은 수자로 세팅해바라, default 50mm
    width_mm = 20
    width_pixel = int(np.round(width_mm / pixel_spacing))
    top_image = slice_img[:min_bottom, int(center_column - width_pixel // 2):int(center_column + width_pixel // 2)]
    top_mean = top_image.mean(axis=1)

    # TODO: 여기서 이 0.52 가 어디서 왔나, 이걸 다른것으로 세팅할 수 없을까
    bone_threshold = 0.51
    # print(top_mean)
    top_mean[top_mean < bone_threshold] = 0
    min_top = top_mean.nonzero()[0][0]
    return min_top, min_slice, slice_img

# def find_center()
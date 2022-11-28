import os
import numpy as np
import nibabel as nib

def read_patient_nii_mask(UID, seg_dir):
    nii_path = os.path.join(seg_dir, f"{UID}.nii")
    seg_mask = nib.load(nii_path).get_fdata()
    seg_mask = np.rot90(seg_mask, axes=(0, 1))
    seg_mask = seg_mask.transpose((2, 0, 1))
    seg_mask = np.flip(seg_mask, axis=0)
    seg_mask[seg_mask > 7] = 0
    return seg_mask

def get_uid_from_niipath(path):
    return path.split("/")[-1].replace(".nii", "")
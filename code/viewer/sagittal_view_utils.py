import os
import glob
import pydicom
import nibabel as nib
import pandas as pd
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from tqdm import tqdm

from PIL import Image, ImageOps



def get_PatientUID(UID):
    return "1.2.826.0.1.3680043." + UID

def get_sagittal_image(data_dir, UID_index):
    UID, index = UID_index.split("-")
    PatientUID = get_PatientUID(UID)
    # print(index)
    image_path = os.path.join(data_dir, PatientUID, f"{int(float(index))}.jpeg")

    img = Image.open(image_path)
    return img

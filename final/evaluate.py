import glob
import os
import re
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
import torchvision
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
import numpy as np


from effdet.bench import _post_process
from effdet.anchors import Anchors, decode_box_outputs

# IMAGES_DIR = '/root/autodl-tmp/cervical_spine/train_axial_images_jpeg95_croped_132508/'
# MASK_DIR = '/root/autodl-tmp/cervical_spine/segmentation_axial_results_132508/'
boundary_df_path = '/root/autodl-tmp/cervical_spine/infered_boundary_132508_2.csv'
train_df_path = '/root/autodl-tmp/cervical_spine/train.csv'

device = 'cuda'

class ImageDataSet(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, mask_dir, transform):
        super().__init__()
        self.df = df
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.len = len(self.df)

    def __getitem__(self, i):

        s = self.df.iloc[i]
        UID = s.StudyInstanceUID

        img_path = os.path.join(self.img_dir, UID, f'{int(s.Slice)}.jpeg')
        img = Image.open(img_path)

        mask_path = os.path.join(self.mask_dir, UID, f'{int(s.Slice)}.png')
        mask = Image.open(mask_path)

        mask = mask.crop((s.xmin/2, s.ymin/2, s.xmax/2, s.ymax/2))

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        return img, mask

    def __len__(self):
        return self.len

class ImageTransform(nn.Module):
    def __init__(self, image_size=640):
        super().__init__()

        self.image_size = image_size

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

        self.mask_transform = T.Compose([
            T.Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def forward(self, x, mask):
        x = self.transform(x)
        mask = self.mask_transform(mask)
        return x, mask

def pred_det_yolo(model, x, anchors=None):
    pred = model(x)[0]
    max_indices = torch.argmax(pred[:, :, 4], dim=1)
    max_values = pred[torch.arange(x.shape[0]), max_indices, :] # N x 6

    bboxes, scores = max_values[:, :4], max_values[:, 4]
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    return bboxes, scores


def decode_model_outputs(class_out, box_out, anchors, num_levels):
    class_out, box_out, indices, classes = _post_process(
                class_out, box_out, num_levels=num_levels, num_classes=1,
                max_detection_points=1)
    anchor_boxes = anchors.boxes[indices, :]
    
    box_list = []
    score_list = []
    
    for i in range(class_out.shape[0]):
        box_outputs = box_out[i]
        boxes = decode_box_outputs(box_outputs.float(), anchor_boxes[i], output_xyxy=True)
        scores = class_out[i].sigmoid()
        
        box_list.append(boxes)
        score_list.append(scores)
    boxes = torch.cat(box_list, axis=0)
    scores = torch.cat(score_list, axis=0).reshape(-1)
    
    return boxes, scores
    # return torch.cat((boxes, scores.sigmoid()), axis=1)


def pred_det_effdet(model, x, anchors):
    x = (x - 0.5) * 2
    class_out, box_out = model(x)
    
    return decode_model_outputs(class_out, box_out, anchors, model.config.num_levels)
    


def get_bbox_class_list(seg_list, seg_bboxes):
    class_list = []
    for i in range(seg_list.shape[0]):
        class_index = get_bbox_class(seg_list[i, :, :], seg_bboxes[i, :])
        class_list.append(class_index)

    return torch.stack(class_list)


def get_bbox_class(seg, bbox):
    """
    label 은 0.125 의 단위로,
    seg: H x W
    bbox: [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, xmax, ymax = bbox.int()
    area = seg[ymin:ymax, xmin:xmax]

    result = torch.mean(area[area>0])
    result = torch.round(result / 0.125)

    return result

def get_class_score(scores, class_list, eps=1e-2):
    result = scores.new_zeros((scores.shape[0], 8)) + eps
    class_list = torch.nan_to_num(class_list).long()
    result[torch.arange(scores.shape[0]), class_list] = scores

    return result


def cal_loss(prob, label):

    pos_weight = np.array([14, 2, 2, 2, 2, 2, 2, 2])
    neg_weight = np.array([7, 1, 1, 1, 1, 1, 1, 1])

    
    pos_score = (-pos_weight * label * np.log(prob)).sum(axis=1)
    neg_score = (-neg_weight * (1 - label) * np.log(1 - prob)).sum(axis=1)
    # print(pos_score)
    # score = pos_weight * label * np.log(prob) + neg_weight * (1 - label) * np.log(1 - prob)

    # weight_total = pos_weight * label + neg_weight * (1 - label)
    pos_weight_total = (pos_weight * label).sum(axis=1)
    neg_weight_total = (neg_weight * (1-label)).sum(axis=1)
    
    
    return (pos_score + neg_score) / (pos_weight_total + neg_weight_total), pos_score / (pos_weight_total + 1e-9), neg_score / (neg_weight_total + 1e-9)
    
    # return -score.sum(axis=1) / weight_total.sum(axis=1)


def get_test_dataloader(df, batch_size=32, image_size=512, image_dir=None, mask_dir=None):
    tf = ImageTransform(image_size=image_size)
    ds = ImageDataSet(df, image_dir, mask_dir, tf)

    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=min(16, batch_size))
    return dl


def get_test_prediction_df(df, predictions):
    df_yolo_pred = pd.DataFrame(
        data=predictions, columns=['patient_overall'] + [f'C{i}' for i in range(1, 8)]
    )
    df_test_pred = pd.concat([df, df_yolo_pred], axis=1).sort_values(['StudyInstanceUID', 'Slice'])
    df_patient_pred = df_test_pred.groupby('StudyInstanceUID').apply(lambda df: df.max())

    clip_value = 1e-3
    df_patient_pred[[f'C{i}' for i in range(1, 8)]] = df_patient_pred[[f'C{i}' for i in range(1, 8)]].clip(lower=clip_value, upper=1-clip_value)


    df_patient_pred["patient_overall"] = df_patient_pred[[f'C{i}' for i in range(1, 8)]].max(axis=1)
    df_patient_pred = df_patient_pred[['patient_overall'] + [f'C{i}' for i in range(1, 8)]]

    return df_patient_pred

def log_test_score(train_df, pred_df, save_path):
    prob = pred_df.values
    label = train_df.loc[pred_df.index].values
    
    losses, pos_losses, neg_losses = cal_loss(prob, label)
    # losses = pos_losses + neg_losses
    
    
    pred_df["losses"] = losses
    pred_df["pos_losses"] = pos_losses
    pred_df["neg_losses"] = neg_losses
    
    pred_df[['label_patient_overall'] + [f'label_C{i}' for i in range(1, 8)]] = label
    pred_df.to_csv(save_path, mode='a', header=None)

    return float(np.mean(losses)), float(np.mean(pos_losses)), float(np.mean(neg_losses))

def test_predict(model, dl, batch_size=32, model_name='effdet', image_size=512, anchors=None):

    pred_det = pred_det_effdet if model_name == 'effdet' else pred_det_yolo
    
    with torch.no_grad():

        predictions = []

        for x, mask in tqdm(dl):
            x = x.to(device)
            # print(x.shape, x.min(), x.max())
            mask = mask.to(device)

            batch_probs = x.new_zeros((x.shape[0], 8)) + 1e-3

            active_indices = mask.sum(axis=[1, 2, 3]).nonzero().reshape(-1)

            if active_indices.numel() == 0:
                predictions.append(batch_probs)
                continue


            if active_indices.numel() != batch_size:
                x = x[active_indices, :, :, :]
                mask = mask[active_indices, :, :, :]

            bboxes, scores = pred_det(model, x, anchors)    
            # print(scores)
            # if model_name == 'effdet':
            #     bboxes, scores = pred_det_effdet(model, x, anchors)
            # else:
            #     bboxes, scores = pred_det_yolo(model, x)

            class_list = get_bbox_class_list(mask[:, 0, :, :], bboxes / (image_size / 256.) )
            probs = get_class_score(scores, class_list)

            batch_probs[active_indices, :] = probs
            predictions.append(batch_probs)

        return torch.concat(predictions).cpu().numpy()


class Evaluate():
    def __init__(self, model, save_dir, image_size, image_dir, mask_dir, log_size=8, model_name='effdet'):
        
        self.log_size = log_size
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.save_dir = save_dir
        self.model_name = model_name
        self.model = model
        self.image_size = image_size
        self.boundary_df = pd.read_csv(boundary_df_path)
        self.train_df = pd.read_csv(train_df_path).set_index('StudyInstanceUID')
        self.UIDs = list(self.boundary_df.StudyInstanceUID.unique())
        
        if model_name == 'effdet':
            self.anchors = Anchors.from_config(model.config).to(device)
    
    def evaluate(self, epoch, batch_size=16):
        
        log_size = self.log_size
        boundary_df = self.boundary_df
        train_df = self.train_df
        UIDs = self.UIDs
        save_dir = self.save_dir
        model = self.model
        
        model.eval()
        
        total_UID_len = len(UIDs)
        start_index = (epoch * log_size) % total_UID_len
        sample_UIDs = UIDs[start_index:(start_index+log_size)]
        
        # for test
        # sample_UIDs = UIDs[6:7]
        # print(sample_UIDs)
        
        df = boundary_df[boundary_df.StudyInstanceUID.isin(sample_UIDs)].reset_index()

        print("test df length : ", len(df))
        dl = get_test_dataloader(df, batch_size=batch_size, image_size=self.image_size, image_dir=self.image_dir, mask_dir=self.mask_dir)
        predictions = test_predict(model, dl, anchors=self.anchors, image_size=self.image_size)
        pred_df = get_test_prediction_df(df, predictions)
        
        loss, pos_loss, neg_loss = log_test_score(train_df, pred_df, save_dir + '_predictions.csv')
        print("loss :{} pos_loss:{} neg_loss:{}".format(loss, pos_loss, neg_loss))
        return loss, pos_loss, neg_loss, start_index
    
    

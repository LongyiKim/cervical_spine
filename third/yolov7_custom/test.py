import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel

import glob
import re
import torch.nn as nn
import pandas as pd
from PIL import Image
import torchvision
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as TF

IMAGES_DIR = '/root/autodl-tmp/cervical_spine/train_axial_images_jpeg95_croped_132508/'
MASK_DIR = '/root/autodl-tmp/cervical_spine/segmentation_axial_results_132508/'
IMAGE_SIZE = 640
# def get_test_df(UIDs, total_boundary_df):
#     """
#     UIDs : 1.2.826.0.1.3680043.219*
#     """
#
#     test_slices = glob.glob(f'{IMAGES_DIR}/{UIDs}/*')
#     test_slices = [re.findall(f'{IMAGES_DIR}/(.*)/(.*).jpeg', s)[0] for s in test_slices]
#
#     df_test_slices = pd.DataFrame(data=test_slices, columns=['StudyInstanceUID', 'Slice'])
#     df_test_slices['UID_Slice'] = df_test_slices['StudyInstanceUID'] + '.' + df_test_slices['Slice'].astype('string')
#     # df_test_slices = df_test_slices.set_index('UID_Slice').astype({'Slice': int})
#     df = total_boundary_df.loc[df_test_slices.UID_Slice].sort_values(['StudyInstanceUID','Slice']).reset_index()
#
#     return df

# def get_test_df(UIDs, boundary_df):
#     return boundary_df[boundary_df.StudyInstanceUID.isin(UIDs)]


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

def pred_det(model, x):
    pred = model(x)[0]
    max_indices = torch.argmax(pred[:, :, 4], dim=1)
    max_values = pred[torch.arange(x.shape[0]), max_indices, :] # N x 6

    bboxes, scores = max_values[:, :4], max_values[:, 4]
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    return bboxes, scores

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

    # print(area)
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


def get_test_dataloader(df, batch_size=32):
    tf = ImageTransform(image_size=640)
    ds = ImageDataSet(df, IMAGES_DIR, MASK_DIR, tf)

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

def test_predict(model, dl, device='cuda', batch_size=32):

    with torch.no_grad():

        predictions = []

        for x, mask in tqdm(dl):
            x = x.to(device)
            mask = mask.to(device)

            batch_probs = x.new_zeros((x.shape[0], 8)) + 1e-2

            active_indices = mask.sum(axis=[1, 2, 3]).nonzero().reshape(-1)

            if active_indices.numel() == 0:
                predictions.append(batch_probs)
                continue


            if active_indices.numel() != batch_size:
                x = x[active_indices, :, :, :]
                mask = mask[active_indices, :, :, :]

            bboxes, scores = pred_det(model, x)

            class_list = get_bbox_class_list(mask[:, 0, :, :], bboxes / (IMAGE_SIZE / 256.) )
            probs = get_class_score(scores, class_list)

            batch_probs[active_indices, :] = probs
            predictions.append(batch_probs)

        return torch.concat(predictions).cpu().numpy()





def test_custom(
        UIDs,
                boundary_df,
                train_df,
                epoch=0,
                log_batch_start=0,
                log_size=8,     # 8 UID per epoch
                weights=None,
                batch_size=32,
                imgsz=640,
                conf_thres=0.001,
                iou_thres=0.6,  # for NMS
                save_json=False,
                single_cls=False,
                augment=False,
                verbose=False,
                model=None,
                dataloader=None,
                save_dir=Path(''),  # for saving images
                save_txt=False,  # for auto-labelling
                save_hybrid=False,  # for hybrid auto-labelling
                save_conf=False,  # save auto-label confidences
                plots=True,
                wandb_logger=None,
                compute_loss=None,
                half_precision=True,
                trace=False,
                is_coco=False,
                v5_metric=False):
    # device = next(model.parameters()).device  # get model device

    # Half
    # half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    # if half:
    #     model.half()

    # Configure
    model.eval()

    total_UID_len = len(UIDs)
    start_index = (epoch * log_size) % total_UID_len
    start_index += log_batch_start
    sample_UIDs = UIDs[start_index:(start_index+log_size)]
    df = boundary_df[boundary_df.StudyInstanceUID.isin(sample_UIDs)].reset_index()

    print("test df length : ", len(df))
    dl = get_test_dataloader(df)
    predictions = test_predict(model, dl)
    pred_df = get_test_prediction_df(df, predictions)
    loss, pos_loss, neg_loss = log_test_score(train_df, pred_df, save_dir / 'predictions.csv')
    print("loss :{} pos_loss:{} neg_loss:{}".format(loss, pos_loss, neg_loss))
    return loss, pos_loss, neg_loss, start_index

def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size
        
        if trace:
            model = TracedModel(model, device, imgsz)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, v5_metric=opt.v5_metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot

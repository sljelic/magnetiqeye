import warnings
import os
import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple

from apds.lib.detection_utils import merge_predictions_by_frame_id
from apds.lib import sampling
from apds.lib.raw_dataset import normalize_frame_id
from apds.lib.detection_utils import io_formats
from apds.lib.datatypes import Detection, VideoId
from apds.stats import lib
from apds.stats.lib import get_boxes_array


def read_prediction_gt_pairs(
    pred_path, gt_path, sample=None, sort=False, Nmax=10000
) -> List[Tuple[Detection, Detection]]:
    if sample is not None:
        sample = set(sampling.load_sample(sample))

    predictions = tqdm(io_formats.get_reader(pred_path), desc="load Predictions")
    if sample is not None:
        predictions = (
            d
            for d in predictions
            if sampling.in_sample(normalize_frame_id(d.frame_id), sample)
            # and d.frame_id.supplier != "NonPolyp"
            # if d.frame_id.video == "ShortVD050672"
        )
    predictions = list(predictions)

    groundtruth = tqdm(io_formats.get_reader(gt_path), desc="load GT")
    if sample is not None:
        groundtruth = (
            d
            for d in groundtruth
            if sampling.in_sample(normalize_frame_id(d.frame_id), sample)
            # if d.frame_id.video == "ShortVD050672"
        )
    groundtruth = list(groundtruth)

    pred_gt_pairs, pred_not_seen, gt_not_seen = merge_predictions_by_frame_id(
        predictions, groundtruth
    )
    if pred_not_seen or gt_not_seen:
        warnings.warn(
            f"Mismatched {len(pred_not_seen)} prediction frames "
            f"and {len(gt_not_seen)} ground-truth frames"
        )
    if sort:
        pred_gt_pairs.sort(key=lambda item: item[0].frame_id)

    return pred_gt_pairs


def group_pred_gt_pairs_by_videos(
    pred_gt_pairs,
) -> Dict[VideoId, List[Tuple[Detection, Detection]]]:
    """
    :param pred_gt_pairs: sequence of (Detection, Detection) pairs - prediction and GT
    :return: dict VideoId->List[(Detection, Detection)].
        order of pairs in output lists is preserved,
        so you can pass sorted list of pairs and expect that output lists will be sorted
    """
    pred_gt_pair_lists = {}
    for det, gt in pred_gt_pairs:
        video_id = det.frame_id.video_id
        if video_id not in pred_gt_pair_lists:
            pred_gt_pair_lists[video_id] = []
        pred_gt_pair_lists[video_id].append((det, gt))
    return pred_gt_pair_lists


def save_evaluation_results(output_directory, scalar_metrics_dict, curves_dict):
    """
    :param output_directory: path to directory (mb not existing yet) where to save data
    :param scalar_metrics_dict: dict str->float
    :param curves_dict: dict str->(curve Y array, curve X array);
        arrays can be numpy or plain python sequences.

    Results are saved under given directory this way:
        "metrics.json" - contains scalar metrics
        "curves" - directory storing one JSON file per curve. Curve names (keys) have
            replaced slashes ("/") into underscores ("_")
            Curve JSON file contains two keys - "x" and "y" with lists of values
    """
    metrics_path = os.path.join(output_directory, "metrics.json")
    curves_path = os.path.join(output_directory, "curves")

    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(curves_path, exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(scalar_metrics_dict, f, indent=4)
    for key, curve in curves_dict.items():
        ys, xs = curve
        save_path = os.path.join(curves_path, key.replace("/", "_") + ".json")
        with open(save_path, "w") as f:
            json.dump({"y": np.array(ys).tolist(), "x": np.array(xs).tolist()}, f)


def split_preds_to_fp_and_tp(predictions, groundtruth, iou_thr, model_th):
    pred_gt_pairs = read_prediction_gt_pairs(predictions, groundtruth, sort=True)
    tp_detections = []
    fp_detections = []
    for frame_idx, (det, gt) in enumerate(pred_gt_pairs):
        det_boxes = []
        for box in det.boxes:
            if box.confidence >= model_th:
                det_boxes.append(box)
        [(tp_flags, match_ids, n_fn)] = lib.match_detections_to_gt(
            [bb.coordinates for bb in det_boxes],
            [bb.coordinates for bb in gt.boxes],
            iou_thresholds=[iou_thr],
        )
        false_boxes = []
        true_boxes = []
        assert len(tp_flags) == len(det_boxes)
        for i in range(len(tp_flags)):
            if tp_flags[i]:
                true_boxes.append(det_boxes[i])
            else:
                false_boxes.append(det_boxes[i])
        tp_detections.append(
            Detection(
                boxes=true_boxes,
                frame_id=det.frame_id,
                pointers=det.pointers,
                contours=det.contours,
            )
        )
        fp_detections.append(
            Detection(
                boxes=false_boxes,
                frame_id=det.frame_id,
                pointers=det.pointers,
                contours=det.contours,
            )
        )
    return tp_detections, fp_detections

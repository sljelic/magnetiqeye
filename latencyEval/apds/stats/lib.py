import numpy as np
import bisect
from typing import Collection


def boxes_iou_matrix(boxes1: "[N1 yxYX]", boxes2: "[N2 yxYX]") -> "[N1 N2]":
    """
    :param boxes1: 2d numpy array of N1 boxes encoded as yxYX
    :param boxes2: 2d numpy array of N2 boxes encoded as yxYX
    :return: 2d matrix of pairwise IoU values
    """
    n1 = len(boxes1)
    n2 = len(boxes2)
    if n1 == 0 or n2 == 0:
        return np.zeros([n1, n2])

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # broadcast both arrays to matrix of common shape
    boxes1 = np.tile(boxes1[:, np.newaxis], [1, n2, 1])
    boxes2 = np.tile(boxes2[np.newaxis, :], [n1, 1, 1])
    area1 = np.tile(area1[:, np.newaxis], [1, n2])
    area2 = np.tile(area2[np.newaxis, :], [n1, 1])

    intersection_ymin = np.maximum(boxes1[:, :, 0], boxes2[:, :, 0])
    intersection_ymax = np.minimum(boxes1[:, :, 2], boxes2[:, :, 2])
    intersection_h = np.maximum(0, intersection_ymax - intersection_ymin)

    intersection_xmin = np.maximum(boxes1[:, :, 1], boxes2[:, :, 1])
    intersection_xmax = np.minimum(boxes1[:, :, 3], boxes2[:, :, 3])
    intersection_w = np.maximum(0, intersection_xmax - intersection_xmin)

    intersection_a = intersection_h * intersection_w
    union_a = area1 + area2 - intersection_a

    return intersection_a / union_a


def auc(curve_ys, curve_xs):
    # assuming X already sorted ascending
    return np.trapz(curve_ys, curve_xs)


class ConfidenceThresholdBasedMetrics:
    def __init__(self, thresholds: "[N_thresholds] float ascending-sorted"):
        self.thresholds = thresholds
        n_thr = len(thresholds)
        self.tp = np.zeros([n_thr], int)
        self.fp = np.zeros([n_thr], int)
        self.n_gt = 0
        self.n_frames = 0
        self._cache = {}

    def add(self, confidences, labels):
        if np.ndim(confidences) == 0:
            confidences = [confidences]
            labels = [labels]
        self.n_frames += 1
        self.n_gt += np.sum(labels)
        positions = np.searchsorted(self.thresholds, confidences)
        for i, is_true in zip(positions, labels):
            if is_true:
                self.tp[:i] += 1
            else:
                self.fp[:i] += 1
        if self._cache:
            self._cache.clear()

    @property
    def froc_curve(self):
        curve = self._cache.get("froc_curve")
        if curve is None:
            tpr = self.tp / max(1, self.n_gt)
            fpr = self.fp / max(1, self.n_frames)
            if fpr[0] < 1.0:
                tpr = np.pad(tpr, [(1, 0)], mode="constant", constant_values=1.0)
                fpr = np.pad(fpr, [(1, 0)], mode="constant", constant_values=1.0)
            curve = tpr[::-1], fpr[::-1]
            self._cache["froc_curve"] = curve
        return curve

    @property
    def froc_auc(self):
        return auc(*self.froc_curve)

    @property
    def pr_curve(self):
        curve = self._cache.get("pr_curve")
        if curve is None:
            if self.n_gt > 0:
                recall = self.tp / self.n_gt
            else:
                recall = np.ones(self.tp.shape, dtype=np.float)
            all_p = self.tp + self.fp
            zero_p = (all_p == 0)
            non_zero_p = ~zero_p
            precision = np.ones(self.tp.shape, dtype=np.float)
            precision[zero_p] = 1
            precision[non_zero_p] = self.tp[non_zero_p] / all_p[non_zero_p]
            recall = np.pad(recall, [(0, 1)], mode="constant", constant_values=0)
            precision = np.pad(precision, [(0, 1)], mode="constant", constant_values=1)
            curve = precision[::-1], recall[::-1]
            self._cache["pr_curve"] = curve
        return curve

    @property
    def pr_auc(self):
        return auc(*self.pr_curve)

    def merge(self, others: Collection['ConfidenceThresholdBasedMetrics']):
        for other in others:
            self.tp += other.tp
            self.fp += other.fp
            self.n_gt += other.n_gt
            self.n_frames += other.n_frames
        self._cache.clear()


def match_detections_to_gt(
    detection_boxes: "[N1 yxYX]", gt_boxes: "[N2 yxYX]", iou_thresholds: "[ N_iou ]"
):
    """
    :return: list of N_iou tuples of : ([N1] bool, [N1] int, int)
        where first item is an array of boolean whether corresponding detection
            is a TP (true) or FP (false), at corresponding level of IoU;
        second item is an array of indices of GTs matched by corresponding detection;
        third item is amount of GTs that was not "covered" by any of detections.
    """
    n_gt = len(gt_boxes)
    if np.size(detection_boxes) == 0:
        return [(np.zeros([0], bool), np.zeros([0], int), n_gt) for _ in iou_thresholds]
    if np.size(gt_boxes) == 0:
        n_dets = len(detection_boxes)
        return [
            (np.zeros([n_dets], bool), np.zeros([n_dets], int), 0)
            for _ in iou_thresholds
        ]
    iou_mat = boxes_iou_matrix(detection_boxes, gt_boxes)
    result = []
    for iou_thr in iou_thresholds:
        tp_mat = np.greater_equal(iou_mat, iou_thr)
        tp_array = np.any(tp_mat, axis=1)
        match_indices = np.argmax(iou_mat, axis=1)
        n_matched = len(set(match_indices[tp_array]))
        n_not_matched = n_gt - n_matched
        result.append((tp_array, match_indices, n_not_matched))
    return result


def get_boxes_array(detection):
    return np.array([bb.coordinates for bb in detection.boxes])


def get_scores_array(detection):
    return np.array([bb.confidence for bb in detection.boxes])


class Evaluator:
    def __init__(self, iou_thresholds, with_1polyp_froc=False):
        self._iou_thresholds = tuple(iou_thresholds)
        self._with_1polyp_froc = with_1polyp_froc

        conf_thresholds = np.linspace(0, 1, num=10000, endpoint=False)
        self._thr_based_metrics = [
            ConfidenceThresholdBasedMetrics(conf_thresholds)
            for _ in self._iou_thresholds
        ]
        if self._with_1polyp_froc:
            self._for_1polyp_froc = [
                ConfidenceThresholdBasedMetrics(conf_thresholds)
                for _ in self._iou_thresholds
            ]
        else:
            self._for_1polyp_froc = []

    def add_boxes_from_one_frame(
        self,
        detection_boxes: "[N1 yxYX]",
        detection_confidences: "[ N1 ]",
        gt_boxes: "[N2 yxYX]",
        confidence_threshold=None,
    ):
        if confidence_threshold is not None:
            mask = np.greater_equal(detection_confidences, confidence_threshold)
            detection_confidences = detection_confidences[mask]
            detection_boxes = detection_boxes[mask]
        match_results = match_detections_to_gt(
            detection_boxes, gt_boxes, self._iou_thresholds
        )

        for iou_i in range(len(self._iou_thresholds)):
            tp_flags, match_ids, n_fn = match_results[iou_i]
            seen = set()
            for i, m in enumerate(match_ids):
                if not tp_flags[i]:
                    continue
                # this relies on detections being sorted by confidence descending:
                # if few boxes intersect the same GT box, only top-confident is TP
                if m not in seen:
                    seen.add(m)
                else:
                    tp_flags[i] = False
            thr_metrics_obj = self._thr_based_metrics[iou_i]
            thr_metrics_obj.add(detection_confidences, tp_flags)
            thr_metrics_obj.n_gt += n_fn

        if self._with_1polyp_froc:
            match_results_1polyp = match_detections_to_gt(
                detection_boxes, gt_boxes[:1], self._iou_thresholds
            )
            n_gt = len(gt_boxes)
            if len(detection_confidences) > 0:
                top1_det_i = np.argmax(detection_confidences)
            else:
                top1_det_i = None

            for iou_i in range(len(self._iou_thresholds)):
                thr_metrics_obj_1polyp = self._for_1polyp_froc[iou_i]
                if top1_det_i is not None:
                    tp_flags_1polyp, _, _ = match_results_1polyp[iou_i]
                    top1_is_tp = tp_flags_1polyp[top1_det_i]
                    thr_metrics_obj_1polyp.add(
                        detection_confidences[top1_det_i], top1_is_tp
                    )
                    has_tp = top1_is_tp
                else:
                    thr_metrics_obj_1polyp.add(0, False)
                    has_tp = False
                if not has_tp:
                    if n_gt > 0:
                        thr_metrics_obj_1polyp.n_gt += 1

    def add_detection_objects_from_one_frame(
        self, detection, gt, confidence_threshold=None
    ):
        det_boxes = get_boxes_array(detection)
        det_scores = get_scores_array(detection)
        gt_boxes = get_boxes_array(gt)
        self.add_boxes_from_one_frame(
            det_boxes, det_scores, gt_boxes, confidence_threshold
        )

    def merge(self, others: Collection['Evaluator']):
        for other in others:
            for meter_own, meter_other in zip(
                self._thr_based_metrics + self._for_1polyp_froc,
                other._thr_based_metrics + other._for_1polyp_froc,
            ):
                meter_own.merge([meter_other])

    def get_metrics_dict(self):
        curves = {}
        scalars = {}
        for iou_i, iou_thr in enumerate(self._iou_thresholds):
            thr_metrics_obj = self._thr_based_metrics[iou_i]

            curves[f"mROC/Curve/IoU={iou_thr:.2f}"] = thr_metrics_obj.froc_curve
            curves[f"PR/Curve/IoU={iou_thr:.2f}"] = thr_metrics_obj.pr_curve

            scalars[f"PR/AUC/IoU={iou_thr:.2f}"] = thr_metrics_obj.pr_auc

            if self._with_1polyp_froc:
                thr_metrics_obj_1p = self._for_1polyp_froc[iou_i]
                curves[f"ROC/Curve/IoU={iou_thr:.2f}"] = thr_metrics_obj_1p.froc_curve
                scalars[f"ROC/AUC/IoU={iou_thr:.2f}"] = thr_metrics_obj_1p.froc_auc

        return curves, scalars

    def get_precision(self, iou, *, recall=None, threshold=None):
        assert (recall, threshold).count(None) == 1
        index = self._iou_thresholds.index(iou)
        precision_arr, recall_arr = self._thr_based_metrics[index].pr_curve
        if threshold is not None:
            prec_i = self._get_index(threshold, recall_arr)
        else:
            prec_i = bisect.bisect_left(recall_arr, recall)
            if prec_i == len(recall_arr):
                if recall_arr[-1] < recall - 1e-5:
                    return 0.0
                prec_i = len(precision_arr) - 1
        return precision_arr[prec_i]

    def get_recall(self, iou, *, precision=None, threshold=None):
        index = self._iou_thresholds.index(iou)
        precision_arr, recall_arr = self._thr_based_metrics[index].pr_curve

        if threshold is None:
            indices = np.where(precision_arr >= precision)[0]
            if len(indices) == 0:
                return 0.0
            return np.max(recall_arr[indices])
        else:
            index = self._get_index(threshold, recall_arr)
            return recall_arr[index]

    def get_threshold(self, iou, *, precision=None, fppf=None):
        index = self._iou_thresholds.index(iou)
        if precision is not None:
            precision_arr, recall_arr = self._thr_based_metrics[index].pr_curve

            indices = np.where(precision_arr >= precision)[0]
            if len(indices) == 0:
                return 1.0
            indices_index = np.argmax(recall_arr[indices])
            result_index = indices[indices_index]
            return 1 - (result_index / len(precision_arr))
        elif fppf is not None:
            tpr, fpr = self._thr_based_metrics[index].froc_curve
            i = np.searchsorted(fpr, fppf)
            i = len(fpr) - i - 1
            thresholds = self._thr_based_metrics[0].thresholds
            i = np.clip(i, 0, len(thresholds)-1)
            return thresholds[i]
        else:
            raise TypeError

    def get_fppf(self, iou, *, threshold):
        index = self._iou_thresholds.index(iou)
        tpr, fpr = self._thr_based_metrics[index].froc_curve
        return fpr[self._get_index(threshold, fpr)]

    def get_n_fp(self, iou, *, threshold):
        index = self._iou_thresholds.index(iou)
        return self._thr_based_metrics[index].fp[self._get_index(1-threshold)]

    def get_n_tp(self, iou, *, threshold):
        index = self._iou_thresholds.index(iou)
        return self._thr_based_metrics[index].tp[self._get_index(1-threshold)]

    def get_n_fn(self, iou, *, threshold):
        return self.get_n_gt() - self.get_n_tp(iou, threshold=threshold)

    def get_n_gt(self):
        return self._thr_based_metrics[0].n_gt

    def get_max_recall_with_precision(self, iou):
        index = self._iou_thresholds.index(iou)
        precision_arr, recall_arr = self._thr_based_metrics[index].pr_curve
        return recall_arr[-1], precision_arr[-1]

    def get_metrics_keys(self):
        curves, scalars = self.get_metrics_dict()
        return list(curves.keys()), list(scalars.keys())

    def _get_index(self, threshold, array=None):
        if array is None:
            array = self._thr_based_metrics[0].thresholds
        index = int(len(array) * (1 - threshold))
        if index == len(array):
            index -= 1
        return index

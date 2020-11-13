
import numpy as np

def drop_out_detections_without_gt(sprs):
    return list(filter(lambda x: len(x[1].boxes) > 0, sprs ))
def filter_detections(x, minConfidenceAllowed, minIOUAllowed = 0.01):
    """

    Parameters
    ----------
    x : (Detection,Detection)
        pair of detection and gt
    minConfidenceAllowed : float
        predictions with smaller confidence than this value are ignored..
    minIOUAllowed : float, optional
        reds and gt with smaller  IOU than this value are not called "detected". The default is 0.01.

    Returns
    -------
    bool
        True if x is relevant detection, False otherwise.
        If there are more then one bounding box in  prediction, the most confident, 
        among all that significantly intersect gt, is selected and if the confidence 
        is at least minConfidenceAllowed, return True, false, otherwise
    """
    pred_detection = x[0]
    gt_detection = x[1]
    
    pred_boxes = pred_detection.boxes
    if (len(gt_detection.boxes) == 0):
        # there are no objects on the frame
        return False
        
    gt_box = gt_detection.boxes[0]
    
    # take only prediction boxes that significantly intersect gt box
    pred_intersect = list(filter(lambda x: gt_box.iou(x) >=minIOUAllowed, pred_boxes))
    
    if len(pred_intersect) == 0:
        # there are no intersecting boxes with gt box
        return False
    else:
        # that the most confident box that intersects with gt
        most_confident = np.argmax(np.array([x.confidence for x in pred_intersect]))
        if pred_boxes[most_confident].confidence >= minConfidenceAllowed:
            return True
        else:
            return False
        
        
def find_objects_in_video(video, minIOUAllowed = 0.01 ):
    """
    Parameters
    ----------
    video : [(Detection, Detection)]
        list of detections in video.
    minIOUAllowed : float, optional
        if IOU of two bounding box in gt is smalller that minIOUAllowed, we consider
        these bounding boxes to cover different objects. The default is 0.01.

    Returns
    -------
    start_obj_frame_ind : TYPE
        start indices of objects in video

    """
    
    # returns start frames of detected objects
    gt_boxes = [x[1].boxes[0] for x in video]
    pairs_of_boxes = zip(gt_boxes,gt_boxes[1:])
    ious = list(map(lambda x: x[0].iou(x[1]), pairs_of_boxes))
    start_obj_frame_ind = [0] # we consider that objects start with video
    for i in range(len(ious)):
        if ious[i] <minIOUAllowed: # when we find that i-th gt bounding box does not sufficiently interset the (i+1)-th
        # we consider that (i+1)-th covers different object
            start_obj_frame_ind.append(i+1)
            # we save it as a start of the next object    
    return start_obj_frame_ind

def find_objects_in_videos(videos, minIOUAllowed = 0.01):
    """
    

    Parameters
    ----------
    videos : dict( (str, str) -> [(Detection, Detection)])
        dictionaries of videos. key sis video name (composed according to predefined function).
        values is a list of detections in this video
    minIOUAllowed : TYPE, optional
        if IOU of two bounding box in gt is smalller that minIOUAllowed, we consider
        these bounding boxes to cover different objects. The default is 0.01.

    Returns
    -------
    objects : dict( (str,str) -> [[(Detection, Detection)]])
        dict that contains a list of objects for each video
        object is a list of (Detection,Detection)

    """
    objects = dict()
    for key in videos:
        objects_in_video = []
        # find starting indices of objects in video[key]
        indices = find_objects_in_video(videos[key], minIOUAllowed )
        
        if len(indices) > 1:
            for j in range(len(indices)-1):
                objects_in_video.append(videos[key][indices[j] : indices[j+1] ])
        else:
            objects_in_video.append(videos[key][indices[-1] : ])
        objects[key] = objects_in_video
    return objects



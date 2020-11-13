
"""
# =============================================================================
#        The code should be at Production Level, including relevant tests 
# =============================================================================

The Goal:
    Your mission is to write a class that calculates object detection latency,
    it should be able to support:
        1. Detection latency per object 
        2. Mean detection latency per video
        3. Max detection latency per video

Inputs:
    sprs: pairs iterable, returns tuples of the form (pred, gt).
    minConfidenceAllowed: float, predictions with smaller 
                          confidence than this value are ignored.
    minIOUAllowed: float, preds and gt with smaller 
                   IOU than this value are not called "detected".

Extra Info:
    
    Correct Detection: 
        an object is detected correctly if two conditions meet:
            1. The object's confidence is greater than minConfidenceAllowed
            2. The IOU of the prediction and the gt is greater than 0.01
        any detection that does not meet these conditions should not be 
        counted, and should be ignored.
        
    Latency: 
        as we describe it, is the amount of frames that passed since,
        an object has been seen for the first time in the ground truth,
        untill it was discovered for the first time in the prediction.
        
    sprs: 
        is an iterator that outputs pairs of predictions (pred) and 
        ground_truths (gt). 
        
        Each returned argument of the pair is of class "Detection" (apds.datatypes)
        and holds the relevant frames information, that is:
            boxes:  
                a list of bounding boxes coordinates [xmin, ymin, xmax, ymax] 
                and their matching prediction confidence [0 <= conf <= 1]
            frame_id: 
                (hospital, vid_id, frame_number)
                
        usage example:
            in:
                for det_pred, det_gt in sprs:
                    print(det_pred)
                    print(det_gt)
                    break
            out:
                Detection(boxes=(<conf: 0.6299969553947449, 
                                 coord: [105.56462860107422, 
                                         637.5382690429688, 
                                         512.47900390625, 
                                         968.6673583984375]>,), 
                          frame_id=('AsuMayoTest_clean', 'testVD13', 1), 
                          pointers=(), contours=None)
                
                Detection(boxes=(<conf: 1.0, 
                                 coord: [65.0, 
                                         679.0, 
                                         471.0, 
                                         930.0]>,), 
                          frame_id=('AsuMayoTest_clean', 'testVD13', 1), 
                          pointers=(), contours=None)

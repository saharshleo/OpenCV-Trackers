from utility import *
from kalman import kalman_tracker
import numpy as np

def iou(detections, estimate) :
    m=len(detections)
    n=len(estimate)
    iou_matrix=[]
    for i in range(m) :
        iou_of_det_with_tracks=[]
        boxA = detections[i,:]
        # compute area of ground truth bounding box
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        for j in range(n) :
            boxB = estimate[j,:]
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            # compute the area of intersection of ground truth and estimated bounding box
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            # compute the area of estimated position bounding box
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou = interArea / float(boxAArea + boxBArea - interArea)
            # return the intersection over union value
            iou_of_det_with_tracks.append(iou)
        iou_matrix.append(iou_of_det_with_tracks)
    return iou_matrix

def associate_detections(detections, tracks, iou_threshold= 0.3) :
    num_dets=len(detections)
    num_tracks=len(tracks)
    if(num_tracks == 0):
        return np.empty((0,2),dtype = int), np.arange(num_dets), np.empty((0,5),dtype = int)
    iou_matrix = iou(detections,tracks)
    matched_indices=np.empty(shape=(0,2),dtype=int)
    if min(num_dets,num_tracks) > 0:
        max_i,max_j=maximum(iou_matrix,num_dets,num_tracks)
        max_iou=iou_matrix[max_i][max_j]
        iou_matrix[max_i][max_j]=-1
        while(max_iou>iou_threshold):
            if((max_i not in matched_indices[:,0]) and (max_j not in matched_indices[:,1])):
                matched_indices = np.append(matched_indices,[[max_i,max_j]],axis=0)
            max_i,max_j=maximum(iou_matrix,num_dets,num_tracks)
            max_iou=iou_matrix[max_i][max_j]
            iou_matrix[max_i][max_j]=-1
   
    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t, trk in enumerate(tracks):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)
    
    return matched_indices, unmatched_detections, unmatched_trackers

class sort :
    def __init__(self,max_age = 2,min_hits = 3,iou_threshold= 0.3) :
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
    
    def update(self, detections=np.empty((0, 5))) :
        self.frame_count += 1
        print("============== FRAME {} ================".format(self.frame_count))
        tracks = np.zeros((len(self.trackers), 5))
        track_for_frame = []
        for i in range(len(tracks)) :
            position=self.trackers[i].predict()[0]
            tracks[i,:]=[position[0], position[1], position[2], position[3], 0]
        
        matched, unmatched_dets, unmatched_trks = associate_detections(detections,tracks, self.iou_threshold)
        
        for i in matched :
            self.trackers[i[1]].update(detections[i[0], :])

        for i in unmatched_dets :
            tracker = kalman_tracker(detections[i,:])
            self.trackers.append(tracker)
        
        for i,tracker in enumerate(self.trackers) :
            state=tracker.get_state()[0]
            if (tracker.age < 1) and (tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                if tracker.id_assigned==False:
                    kalman_tracker.count+=1
                    tracker.id=kalman_tracker.count
                    tracker.id_assigned=True
                    # print("{} CREATED".format(tracker.id))
                track_for_frame.append(np.concatenate((state,[tracker.id+1])).reshape(1,-1))
            if (tracker.age > self.max_age) :
                # print("{} DELETED".format(tracker.id))
                self.trackers.pop(i)

        print("MATCHED:\n{}\n=======\nUNMATCHED DETECTIONS\n{}\n======\nUNMATCHED TRACKERS:\n{}".format(matched,unmatched_dets,unmatched_trks))
        if len(track_for_frame) > 0:
            return np.concatenate(track_for_frame)
        else:
            return np.empty((0,5))

        


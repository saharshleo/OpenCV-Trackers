import numpy as np
from utility import*

class kalman_tracker():
    count=0
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.dim_x=7 # [x,y,area(s),aspect ratio(r),x_vel,y_vel, area_vel]
        self.dim_z=4 # [x,y,s,r]
        # measurement uncertainty/noise   ndarray (dim_z, dim_z), default eye(dim_x)
        self.R = np.eye(self.dim_z,self.dim_z)
        self.R[2:,2:] *= 10.
        # covariance matrix   ndarray (dim_x, dim_x), default eye(dim_x)
        self.P = np.eye(self.dim_x,self.dim_x)
        self.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.P *= 10.
        # Process uncertainty/noise  ndarray (dim_x, dim_x), default eye(dim_x)
        self.Q = np.eye(self.dim_x,self.dim_x)
        self.Q[-1,-1] *= 0.01
        self.Q[4:,4:] *= 0.01
        # filter state estimate  ndarray (dim_x, 1), default = [0,0,0…0]
        self.x = np.zeros((self.dim_x,1))
        self.x[:4] = convert_bbox_to_z(bbox)
        # state transistion matrix  ndarray (dim_x, dim_x)
        self.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        # measurement function   ndarray (dim_z, dim_x)
        self.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        # age denotes for how many frames the object we are tracking is lost
        self.z = convert_bbox_to_z(bbox)
        self.age=0
        self.hit_streak=0
        self.id_assigned=False

    def update(self,bbox):
        self.age = 0
        self.hit_streak += 1
        self.z = convert_bbox_to_z(bbox)
        self.KG = np.matmul(np.matmul(self.P,np.transpose(self.H)),np.linalg.inv(np.add(np.matmul(np.matmul(self.H,self.P),np.transpose(self.H)),self.R)))  
        self.x = np.add(self.x,np.matmul(self.KG,np.subtract(self.z,np.matmul(self.H,self.x))))
        self.P = np.subtract(self.P,np.matmul(self.KG,np.matmul(self.H,self.P)))


    def predict(self):
        self.x = np.matmul(self.F,self.x)
        self.P = np.matmul(np.matmul(self.F,self.P),np.transpose(self.F))
        self.age += 1
        if (self.age>1) :
            self.hit_streak=0
        prediction = convert_x_to_bbox(self.x)
        return prediction

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.x)


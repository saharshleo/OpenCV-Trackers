import numpy as np


class Boosting:

    def __init__(frame,roi,N,F,S,R):
        self.frame = frame
        self.roi = roi
        self.N = N
        self.S = S
        self.R = R


    def build_features(self):
        pass

    def train_weak_classifier(self):
        pass

    def init_sample_weights(self):
        pass

    def init_selector_pool(self):
        pass

    def get_strong_classifier(self):
        pass

    def get_confidence_map(self):
        pass

    def get_bbox(self):
        pass

    def update_strong_classifier(self):
        pass
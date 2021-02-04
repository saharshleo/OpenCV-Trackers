import numpy as np
import random
from FeatureHaar import*
from skimage.transform import integral_image

class Boosting:
    weights_of_sample=[]
    X=[]        #collection of samples
    Y=[]        #labels corresponding sample
    p=0         #total number of positive samples
    n=0         #total number of negative samples
    # consists of index of the selected features from the self.features
    strong_classifier_index=[]
    alphas_for_strong_clf=[]
    ii_image = np.array([])             #integral image of roi
    # contains index of feature in self.features list
    selector_pool = []                #selectors
    ii_search_region = np.array([])   #integral image of the search region


    def __init__(self,frame,roi,N,F,S,R):
        self.frame = frame
        self.roi = roi  #starting and ending cordinate of roi image
        self.F = F      #Number of features
        self.N = N      #Number of samples
        self.S = S      #Number of selectors
        self.R = R      #Number of features to be replaced
        self.features = []      #feature pool
        self.roi_image = frame[int(self.roi[1]):int(self.roi[3])+1,int(self.roi[0]):int(self.roi[2])+1] #getting roi image from roi
        Boosting.ii_image = integral_image(self.roi_image)  #integral image of roi
        self.search_region=[]       #starting and ending cordinates of the search region
    
    #get search region cordinates based on the roi cordinates provided
    def get_search_region(self):
        roi_height = self.roi_image.shape[0]
        roi_width = self.roi_image.shape[1]
        
        #find the starting x of the search region, 0 if out of bound
        self.search_region.append(int(self.roi[0]-roi_width/2))
        self.search_region[0] = 0 if self.search_region[0]<0 else self.search_region[0]

        #find the starting y of the search region, 0 if out of bound
        self.search_region.append(int(self.roi[1]- roi_height/2))
        self.search_region[1] = 0 if self.search_region[1]<0 else self.search_region[1]
        
        #finding the ending x of the search region, max width of image if out of bound
        self.search_region.append(int(self.roi[2]+roi_width/2))
        self.search_region[2] = self.frame.shape[1]-1 if self.search_region[2]>=self.frame.shape[1] else self.search_region[2]

        #finding the ending y of the search region, max height of the image if out of bound
        self.search_region.append(int(self.roi[3]+roi_height/2))
        self.search_region[3] = self.frame.shape[0]-1 if self.search_region[3]>=self.frame.shape[0] else self.search_region[3]

    # extract and find the integral image of search region
    def set_ii_searchregion(self):
        Boosting.ii_search_region = integral_image(self.frame[int(self.search_region[1]):int(self.search_region[3])+1,int(self.search_region[0]):int(self.search_region[2])+1])

    # generates a feature pool
    def build_features(self):
        for i in range(self.F):
            #Creating a feature object
            feature = FeatureHaar()
            #creating a random feature and appending into the list
            feature.generateRandomFeature(self.ii_image,self.ii_image.shape[0],self.ii_image.shape[1])
            self.features.append(feature)

    def train_weak_classifier(self):
        # self.N is the number of sample pixels we want to take into consideration 
        for i in range(self.N):
            # initialize the label of the sample to be picked up to -1(i.e. negative)
            label=-1
            # picking up random coordinates from within the search region defined
            # here x is a coordinate picked at random within the width of the search_region
            x=random.randint(self.search_region[0],self.search_region[2])
            # here y is a coordinate picked at random within the height of the search_region
            y=random.randint(self.search_region[1],self.search_region[3])
            # appending the picked coordinates in the sample list
            (Boosting.X).append([x,y])
            Boosting.weights_of_sample.append(1)
            # checking if the picked up sample lies in the object of interest i.e. roi 
            # if it lies in the roi then we change its corresponding label to 1
            if (self.roi[0]<=x<=self.roi[2]) and (self.roi[1]<=y<=self.roi[3]):
                label=1
            (Boosting.Y).append(label)
            # evaluating all the randomly picked up features on the coordinated picked up
            for j in range(0,self.F):
                feature=self.features[j]
                # if the feature can be applied on the current pixel only then we can evaluate it 
                # we include the features value on the current pixel in the calculation of mu_plus and mu_minus only if the feature can be applied on the pixel 
                if feature.validate_feature_at(x,y,self.search_region):
                    (feature.value_at_sample_pixels).append(feature.evaluate_feature_at(Boosting.ii_search_region,x,y))
                    if label==1:
                        # for every corresponding feature 'p' is the number of positive labelled pixels on which it can be applied 
                        feature.p+=1
                        feature.mu_plus+=feature.value_at_sample_pixels[i]
                    if label==-1:
                        # for every corresponding feature 'n' is the number of negative labelled pixels on which it can be applied
                        feature.n+=1
                        feature.mu_minus+=feature.value_at_sample_pixels[i]
                else:
                    # if a feature is not valid at a certain pixel None is appended  
                    (feature.value_at_sample_pixels).append(None)
                self.features[j]=feature
        for i in range(0,self.F):
            feature=self.features[i]
            # for every feature mu_plus and mu_minus are calculated
            # dividing by zero avoided 
            if(feature.p!=0):
                feature.mu_plus=feature.mu_plus/(2*feature.p)
            if(feature.n!=0):
                feature.mu_minus=feature.mu_minus/(2*feature.n)
            # for every feature threshold is set to the mean of mu_plus and mu_minus
            feature.threshold=(feature.mu_plus+feature.mu_minus)/2
            # if mu_plus >= mu_minus it means that if the value at a particular pixel
            # is greater than the threshold it lies in the mu_plus region and hence it should be classified as 1
            # is less than the threshold it lies in the mu_minus region ad hence it should be classified as -1
            # so if mu_plus>=mu_minus then samples to the right of threshold on the numberline should be classified as 1 hence polarity 1
            # if mu_plus<mu_minus then samples to the left of the threshold on the numberline should be classified as -1 hence polarity -1 as direction is reversed
            if feature.mu_plus>=feature.mu_minus:
                feature.polarity=1
            else:
                feature.polarity=-1  
            self.features[i]=feature
            # print('******Feature {}*****'.format(i)) 
            # print('FEATURE TYPE')
            # print(feature.featureType)
            # print('FEATURE LOCATION')
            # print(feature.location)
            # print('SAMPLE PIXELS')
            # print(Boosting.X)
            # print('LABELS')
            # print(Boosting.Y)
            # print('VALUE AT SAMPLE PIXELS')
            # print(feature.value_at_sample_pixels)
            # print('MU_PLUS')
            # print(feature.mu_plus)
            # print('MU_MINUS')
            # print(feature.mu_minus)
            # print('THRESHOLD')
            # print(feature.threshold)
            # print('POLARITY')
            # print(feature.polarity)      
                            
    #intialise the selectors with random features
    def init_selector_pool(self):
        RandIdx = list(range(0,self.F))
        random.shuffle(RandIdx)
        n = int(self.F/self.S)      #number features to be assign per selector
        for i in range(self.S):
            Boosting.selector_pool.append(list(RandIdx[i*n:(i+1)*n]))

                    
    # after evaluating feature value at all sample pixels and threshold,polarity,etc for a particular feature
    # we have to classify all the sample pixels according to the evaluated boundaries
    def classify_sample_pixels_of_a_feature(self,feature_idx):
        for i in range(len(Boosting.Y)):
            # if the feature was not valid for a particular pixel value at the pixel was set to 'None' previously
            # so if the feature is not valid at a particular pixel, the pixel is forcibly misclassified for that particular feature
            if self.features[feature_idx].value_at_sample_pixels[i] is None:
                self.features[feature_idx].clf_out[i]=-1*Boosting.Y[i]
            # feature classified according to the evaluated boundaries
            elif (self.features[feature_idx].polarity*self.features[feature_idx].value_at_sample_pixels[i]>=self.features[feature_idx].polarity*self.features[feature_idx].threshold):
                self.features[feature_idx].clf_out[i]=1
            else:
                self.features[feature_idx].clf_out[i]=-1
    
    # computing error of a feature on the sample pixels
    def compute_weighted_error_of_a_feature(self,feature_idx):
        error=0
        lamda_wrong=1
        lamda_right=1
        for i in range(len(Boosting.Y)):
            if self.features[feature_idx].clf_out[i]*Boosting.Y[i]==-1:
                lamda_wrong+=Boosting.weights_of_sample[i]
            if self.features[feature_idx].clf_out[i]*Boosting.Y[i]==1:
                lamda_right+=Boosting.weights_of_sample[i]
        error=lamda_wrong/(lamda_wrong+lamda_right)
        return error
   
    def select_best_weakclf_from_selector_and_compute_say_and_update_wt(self,selector):
        n=len(selector)
        alpha=0
        min_error=self.compute_weighted_error_of_a_feature(self.features[selector[0]])
        best_feature_index=0
        # iterating over all the weak classifiers in the selector pool and selecting the one with minimum error  
        for i in range(1,n):
            feature_idx=selector[i]
            error=self.compute_weighted_error_of_a_feature(self.features[feature_idx])
            if error<min_error:
                best_feature_index=feature_idx
                min_error=error
        if min_error>0.5 or min_error==0:
            alpha=0
            return selector[best_feature_index],alpha
        alpha=(1/2)*np.log((1-min_error)/min_error)
        for i in range(self.N):
            if self.features[selector[best_feature_index]].value_at_sample_pixels[i]*Boosting.Y[i]==-1:
                Boosting.weights_of_sample[i]=Boosting.weights_of_sample[i]/(2*min_error)
            if self.features[selector[best_feature_index]].value_at_sample_pixels[i]*Boosting.Y[i]==1:
                Boosting.weights_of_sample[i]=Boosting.weights_of_sample[i]/(2*(1-min_error))
        return selector[best_feature_index],alpha            

    # iterating over selector pool and computing the best weak classifier from the pool and updating the weights of the sample 
    def get_strong_classifier(self):
        for i in range(self.S):
            best_idx,alpha=self.select_best_weakclf_from_selector_and_compute_say_and_update_wt(self.selector_pool[i])
            Boosting.strong_classifier_index.append(best_idx)
            Boosting.alphas_for_strong_clf.append(alpha)

    def get_confidence_map(self):
        pass

    def get_bbox(self):
        pass

    def update_strong_classifier(self):
        pass

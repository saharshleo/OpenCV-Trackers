import random
import math
import numpy as np


class FeatureHaar:
    
    #List of Haar like feature type
    FeatureTypes = ["type-2-x","type-2-y","type-3-x","type-3-y","type-4"]

    def __init__(self):
        self.value = 0
        self.location = []
        self.selector = -1
         
        
    def generateRandomFeature(self, height, width):
        
        valid = False
        minArea = 1 # minimum area of the haar like feature

        while not valid:
            
            #position the start co ordinate of the haar like feature which is choosen randomly
            position = {"row": int(random.randrange(height)),"col":int(random.randrange(width))}

            #Assume haar feature of type-2-x which contain a two rectangle one for light and other for dark 
            #baseDim randomly calculate and store the width and height of one rectangle since both rectangle have same
            #dimension
            #reference https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_haar.html
            baseDim = {"height":int((1-math.sqrt(1-random.random()))*height),"width":int((1-math.sqrt(1-random.random()))*width)}

            #Randomly choosing the a haarFeature and storing in variable
            self.featureType = FeatureHaar.FeatureTypes[random.randrange(len(FeatureHaar.FeatureTypes))]
            
            if self.featureType == "type-2-x":
                # featureHeight ---> number of rectangles in the kernel along height
                # featureWidth ---> number of rectangles in the kernel along width
                featureHeight = 1
                featureWidth = 2

                # check the randomly calculated kernel is valid in ROI or not if not valid above process will be repeated
                if position["row"]+baseDim["height"]*featureHeight >= height or position["col"] + baseDim["width"]*featureWidth>=width:
                    continue

                #check whether the randomly calculated kernel area greater than minArea else above process will be repeated
                area = baseDim["height"]*featureHeight*baseDim["width"]*featureWidth
                if area<minArea:
                    continue

                # Store the location of the Kernal applied in the ROI
                # assuming type-2-x kernel it have 2 rectangles one white and other dark
                # location stores the start co ordinate row, col and dimensions height, width of all rectangles

                self.location = [[position["row"],position["col"],baseDim["height"],baseDim["width"]],
                                 [position["row"],position["col"]+baseDim["width"],baseDim["height"],baseDim["width"]]]

                # teminate the loop                 
                valid = True

            elif self.featureType == "type-2-y":
                featureHeight = 2
                featureWidth = 1
                if position["row"]+baseDim["height"]*featureHeight >= height or position["col"] + baseDim["width"]*featureWidth>=width:
                    continue
                area = baseDim["height"]*featureHeight*baseDim["width"]*featureWidth
                if area<minArea:
                    continue
                self.location = [[position["row"],position["col"],baseDim["height"],baseDim["width"]],
                                 [position["row"]+baseDim["height"],position["col"],baseDim["height"],baseDim["width"]]]
                valid = True

            elif self.featureType == "type-3-x":
                featureHeight = 1
                featureWidth = 3
                if position["row"]+baseDim["height"]*featureHeight >= height or position["col"] + baseDim["width"]*featureWidth>=width:
                    continue
                area = baseDim["height"]*featureHeight*baseDim["width"]*featureWidth
                if area<minArea:
                    continue
                self.location = [[position["row"],position["col"],baseDim["height"],baseDim["width"]],
                                 [position["row"],position["col"]+baseDim["width"],baseDim["height"],baseDim["width"]],
                                 [position["row"],position["col"]+2*baseDim["width"],baseDim["height"],baseDim["width"]]]
                valid = True

            elif self.featureType == "type-3-y":
                featureHeight = 3
                featureWidth = 1
                if position["row"]+baseDim["height"]*featureHeight >= height or position["col"] + baseDim["width"]*featureWidth>=width:
                    continue
                area = baseDim["height"]*featureHeight*baseDim["width"]*featureWidth
                if area<minArea:
                    continue
                self.location = [[position["row"],position["col"],baseDim["height"],baseDim["width"]],
                                 [position["row"]+baseDim["height"],position["col"],baseDim["height"],baseDim["width"]],
                                 [position["row"]+2*baseDim["height"],position["col"],baseDim["height"],baseDim["width"]]]
                valid = True

            elif self.featureType == "type-4":
                featureHeight = 2
                featureWidth = 2
                if position["row"]+baseDim["height"]*featureHeight >= height or position["col"] + baseDim["width"]*featureWidth>=width:
                    continue
                area = baseDim["height"]*featureHeight*baseDim["width"]*featureWidth
                if area<minArea:
                    continue
                self.location = [[position["row"],position["col"],baseDim["height"],baseDim["width"]],
                                 [position["row"],position["col"]+baseDim["width"],baseDim["height"],baseDim["width"]],
                                 [position["row"]+baseDim["height"],position["col"],baseDim["height"],baseDim["width"]],
                                 [position["row"]+baseDim["height"],position["col"]+baseDim["width"],baseDim["height"],baseDim["width"]]]
                valid = True
            
            #calculating the value of haar like feature
            # self.value = haar_feature(integralImage,self.featureType,self.location)


def compute_haar_feature(img_integral, feature_type, feature_coord):
    # padding the integral image to the left and above
    result=np.zeros((img_integral.shape[0]+1,img_integral.shape[1]+1))
    result[1:img_integral.shape[0]+1,1:img_integral.shape[1]+1] = img_integral
    img_integral=result
    # print('PADDED IMAGE INTEGRAL')
    # print(img_integral)
    # getting width and height of feature from feature_coord
    width=feature_coord[0][3]-1
    height=feature_coord[0][2]-1
    # splitting feature to access the numeral in the feature e.g.- 3 in type-3-y for limiting the iterations 
    n=int((feature_type.split('-'))[1])
    # storing value of value of each box of a single feature eg- of 3 boxes in type-3-x
    list_of_val_of_each_rec=[]
    # final value of each feature
    haar_feature_val=0
    for i in range(n):
        # coordinate of top left point of a box f a featu
        coord_list=[feature_coord[i][0],feature_coord[i][1]]
        # corresponding points in the padded integral image...not directly mapped
        # reference https://datasciencechalktalk.com/2019/07/16/haar-cascade-integral-image/
        A=img_integral[coord_list[0]][coord_list[1]]
        B=img_integral[coord_list[0]][coord_list[1]+width+1]    
        C=img_integral[coord_list[0]+height+1][coord_list[1]]
        D=img_integral[coord_list[0]+height+1][coord_list[1]+width+1]  
        #print("A{} B{} C{} D{}".format(A,B,C,D))          
        rect_val=A+D-B-C
        list_of_val_of_each_rec.append(rect_val)
        # alternately add or subtract value of box of a feature
        haar_feature_val+=(math.pow(-1,i)*rect_val)
    #print(list_of_val_of_each_rec)
    return haar_feature_val
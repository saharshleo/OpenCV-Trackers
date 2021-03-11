import random
import math
from skimage.transform import integral_image
import numpy as np

def haar_feature(img_integral,feature_type,feature_coord):
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
        # coordinate of top left point of a box of a feature
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


class FeatureHaar:
    
    #List of Haar like feature type
    FeatureTypes = ["type-2-x","type-2-y","type-3-x","type-3-y","type-4"]

    def __init__(self):
        self.value = 0
        self.location = []
        self.selector=-1
        self.value_at_sample_pixels=[] 
        self.mu_plus = 0
        self.mu_minus = 0
        self.threshold = 0
        self.polarity = 0
        self.clf_out=[]
        self.p=0
        self.n=0
        
    def generateRandomFeature(self,height,width):
        
        valid = False
        minArea =1 #minimum area of the haar like feature

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

    #calculate the coordinates of the respective feature type given the starting cordinate x,y
    def get_feature_coord(self,x,y):
        base_height = self.location[0][2]   
        base_width = self.location[0][3]
        feature_cord = []
        if self.featureType == 'type-2-x':
            feature_cord = [[y,x,base_height,base_width],
                            [y,x+base_width]]
        elif self.featureType == 'type-2-y':
            feature_cord = [[y,x,base_height,base_width],
                            [y+base_height,x]]
        elif self.featureType == 'type-3-x':
            feature_cord = [[y,x,base_height,base_width],
                            [y,x+base_width],
                            [y,x+2*base_width]]
        elif self.featureType == 'type-3-y':
            feature_cord = [[y,x,base_height,base_width],
                            [y+base_height,x],
                            [y+2*base_height,x]]
        elif self.featureType == 'type-4':
            feature_cord = [[y,x,base_height,base_width],
                            [y,x+base_width],
                            [y+base_height,x+base_width],
                            [y+base_height,x]]

        return feature_cord

    # evaluate a feature at a particular pixel
    def evaluate_feature_at(self,ii_image,x,y):
        feature_cord = self.get_feature_coord(x,y)
        value = haar_feature(ii_image,self.featureType,feature_cord)
        return value
            
    # check if a feature can be applied at a particular pixel 
    def validate_feature_at(self,x,y,search_region):
        n = int(self.featureType.split('-')[1])
        feature_cord = self.get_feature_coord(x,y)
        # print(self.featureType)
        # print(feature_cord)
        if n == 4:
            end_x = feature_cord[n-2][1]+feature_cord[0][3]-1
            end_y = feature_cord[n-2][0]+feature_cord[0][2]-1
        else:
            end_x = feature_cord[n-1][1]+feature_cord[0][3]-1
            end_y = feature_cord[n-1][0]+feature_cord[0][2]-1
        # print("endx: {} endy: {}".format(end_x,end_y))
        # print(search_region)
        if end_x <= search_region[2] and end_y <= search_region[3]:
            return True
        return False 


#<------------------ Example Code ------------------------------>

# img = np.array([[ 1 ,2  ,3  ,4  ,5  ,6],
#  [ 2  ,4  ,6  ,8 ,10 ,12],
#  [ 3  ,6  ,9 ,12 ,15 ,18],
#  [ 4  ,8 ,12 ,16 ,20 ,24],
#  [ 5 ,10 ,15 ,20 ,25 ,30],
#  [ 6 ,12 ,18 ,24 ,30 ,36]])
# print('IMAGE:')
# print(img)

# print('IMAGE DIMENSIONS')
# print('{} * {}'.format(img.shape[0],img.shape[1]))

# img_ii = integral_image(img)
# print(img_ii)

# f1  = FeatureHaar()
# f1.generateRandomFeature(img_ii,img.shape[0],img.shape[1])
# print(f1.featureType)
# print(f1.location)
# print(f1.evaluate_feature_at(img_ii,0,0))



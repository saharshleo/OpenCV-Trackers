import numpy as np
import cv2
 

def im2c(img, w2c, color):
    
    color_values = np.array([[0,0,0] , [0,0,1] , [.5 ,.4, .25] , [.5, .5, .5] , [0, 1, 0] , [1, .8, 0] , [1, .5, 1] , [1, 0, 1] , [1, 0, 0] , [1, 1, 1 ] , [ 1, 1, 0 ]])

    RR = img[:,:,2]
    GG = img[:,:,1]
    BB = img[:,:,0]

    index_im = 1 + (RR[:]/8) + 32*(GG[:]/8) + 32*32*(BB[:]/8)

    if color == 0:
        [max1, w2cM] = max(w2c,[],2)
        out = np.reshape(w2cM[index_im[:]],(img.shape[0],img.shape[1]))
        return out

    elif color == -1:
        out = img
        [max1, w2cM] = max(w2c,[],2)
        out2 = np.reshape(w2cM[index_im[:]],(img.shape[0],img.shape[1]))

        for j in range(0,img.shape[0]):
            for i in range(0, img.shape[1]):
                out[j,i,:] = color_values[out2[j,i]] * 255

        return out

    elif color>0 and color < 12: 
        w2cM = w2c[:,color]
        out = np.reshape(w2cM[index_im[:]],(img.shape[0],img.shape[1]))
        return out
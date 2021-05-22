import cv2
from skimage.feature import hog
from skimage.transform import resize
from skimage import exposure
import matplotlib.pyplot as plt

img = cv2.imread('OpenCV-Trackers/assets/dog_test.png')
roi = cv2.selectROI(img)
x,y,w,h = roi
roi_img = img[y:y+h,x:x+w]
r_img = resize(roi_img[:,:,0],(128,64))
g_img = resize(roi_img[:,:,1],(128,64))
b_img = resize(roi_img[:,:,2],(128,64))

fig, (ax) = plt.subplots(3, 2, sharex=True, sharey=True)

#---------------------------------------------------------------------------------
fd, hog_image = hog(roi_img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True)

print('hog shape',hog_image.shape)
print('hog fd',len(fd))

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax[0,0].axis('off')
ax[0,0].imshow(roi_img,cmap = plt.cm.gray)
ax[0,0].set_title('roi')

ax[0,1].axis('off')
ax[0,1].imshow(hog_image_rescaled,cmap = plt.cm.gray)
ax[0,1].set_title('HOG_r')

#---------------------------------------------------------------------------------

fd, hog_image = hog(g_img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax[1,0].axis('off')
ax[1,0].imshow(g_img,cmap = plt.cm.gray)
ax[1,0].set_title('roi')

ax[1,1].axis('off')
ax[1,1].imshow(hog_image_rescaled,cmap = plt.cm.gray)
ax[1,1].set_title('HOG_g')

#---------------------------------------------------------------------------------

fd, hog_image = hog(b_img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax[2,0].axis('off')
ax[2,0].imshow(b_img,cmap = plt.cm.gray)
ax[2,0].set_title('roi')

ax[2,1].axis('off')
ax[2,1].imshow(hog_image_rescaled,cmap = plt.cm.gray)
ax[2,1].set_title('HOG_b')


print(hog_image.shape)
plt.show()

cv2.waitKey(0)
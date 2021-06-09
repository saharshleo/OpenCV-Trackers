'''
Discretizes colors to eleven color names as described in

    Learning Color Names for Real-World Applications
    J. van de Weijer, C. Schmid, J. Verbeek, D. Larlus.
    IEEE Transactions in Image Processing 2009.

The work above uses color names observed in the wild (on Google Images)
to derive probabilities for eleven color names for every point in RGB space.

The function provided here simply assigns each pixel according to its
maximum probability label based on probabilities from van de Weijer.
'''

from scipy.io import loadmat
import numpy
import cv2

# The 11 fundamental named colors in English.
color_names = [
        'black',
        'blue',
        'brown',
        'grey',
        'green',
        'orange',
        'pink',
        'purple',
        'red',
        'white',
        'yellow']

color_values = numpy.array([
        [0, 0, 0],          # black
        [64, 64, 255],      # blue
        [127, 101, 63],     # brown
        [127, 127, 127],    # grey
        [0, 192, 0],        # green
        [255, 203, 0],      # orange
        [192, 64, 192],     # pink
        [255, 64, 255],     # purple
        [255, 0, 0],        # red
        [255, 255, 255],    # white
        [255, 255, 0]],     # yellow
        dtype='uint8')

# Flat weighted mean of RGB values, generated from van de Weijer's data via:
# w2c = numpy.swapaxes(scipy.io.loadmat(
#    'w2c.mat')['w2c'].reshape(32,32,32,11), 0, 2)
# r = numpy.tile(numpy.arange(0,256,8), (32,32,1))
# rr = numpy.stack([numpy.swapaxes(r, 2, n)
#    for n in range(3)])[:,:,:,:,None]
# ((w2c[None] * rr).sum(axis=(1,2,3)) /
#     w2c.sum(axis=(0,1,2))[None]).transpose().astype('uint8')
# What we actually need is weightings by real
# color ocurrences.
mean_color_values = numpy.array(
      [[ 97, 107, 113],     # black
       [ 64, 119, 180],     # blue
       [140, 116,  88],     # brown
       [114, 125, 125],     # grey
       [ 84, 186,  90],     # green
       [182, 106,  65],     # orange
       [188,  90, 136],     # pink
       [138,  65, 172],     # purple
       [155,  84,  77],     # red
       [130, 166, 170],
       [162, 173,  75]],
       dtype='uint8')

# Load
_cached_w2color = None
def get_color_map():
    global _cached_w2color
    if _cached_w2color is None:
        w2c = loadmat('w2c.mat')['w2c']
        _cached_w2color = numpy.swapaxes(
                w2c.argmax(axis=1).reshape(32, 32, 32), 0, 2)
        # numpy.save('w2color', _cached_w2color, False)
        # instead just load this from the saved .npy file
        # from inspect import getsourcefile
        # import os.path
        # _cached_w2color = numpy.load(os.path.join(
        #     os.path.dirname(getsourcefile(lambda:0)), 'w2color.npy'))
    return _cached_w2color

def label_colors(im):
    try:
        if len(im.shape) == 2:
            im = numpy.repeat(im[:,:,numpy.newaxis], 3, axis=2)
        return get_color_map()[tuple(im[:,:,c] // 8 for c in range(3))]
    except:
        print(im.shape)
        print((im // 8).max())
        raise

def label_major_colors(im, threshold=0):
    raw = label_colors(im)
    if not threshold:
        return raw
    pixel_threshold = int(im.shape[0] * im.shape[1] * threshold)
    counts = numpy.bincount(raw.ravel(), minlength=11)
    mask = (counts > pixel_threshold)
    return (raw + 1).astype('int') * mask[raw] - 1

def posterize(im, use_mean_colors=False):
    if use_mean_colors:
         return mean_color_values[label_colors(im)]
    else:
         return color_values[label_colors(im)]


def extract_cn_feature_byw2c(patch):
    w2c = loadmat('w2c.mat')['w2c']
    w2c = numpy.swapaxes(w2c,0,1)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(numpy.float32) / 255 - 0.5
    gray = gray[:, :, numpy.newaxis]

    if numpy.all(patch[:,:,0]==patch[:,:,1]) and numpy.all(patch[:,:,0]==patch[:,:,2]):
        return gray

    b, g, r = cv2.split(patch)
    index_im = ( r//8 + 32 * g//8 + 32 * 32 * b//8)
    h, w = patch.shape[:2]
    w2c=numpy.array(w2c)
  
    out=w2c.T[index_im.flatten(order='F')].reshape((h,w,w2c.shape[0]))

    out=numpy.concatenate((gray,out),axis=2)
    return out


# if __name__ == '__main__':
#     img = cv2.imread('../assets/dog_test.png')
#     cv2.imshow('image',img)
#     roi = cv2.selectROI('roi selector',img)
#     x,y,w,h = roi

#     roi_img = img[y:y+h,x:x+w]

#     # print('image dimensions', img.shape)
#     # result = label_colors(img)
#     # poster = posterize(img)
#     # print('result',result)
#     # print('poster',poster.shape,type(poster))

#     result = extract_cn_feature_byw2c(roi_img)

#     print('image dim',roi_img.shape)
#     print('result shape',result.shape)

#     cv2.imshow('test',result[:,:,7])

#     # cv2.imshow('poster',poster)
#     if cv2.waitKey(0) == ord('x'):
#         cv2.destroyAllWindows()

# KCF

### Algorithm

**main**
* videocap, roi capture
* tracker = KCF.initialize(frame)
* LOOP until stopped:
    * frame = cap.read()
    * getBoundingBox = tracker.update(frame)
    * imshow(frame, getBoundingBox)

**KCF**
* init():
    * declare parameters and initialize object
    * get features from the image (HOG or colored)

* dft():
    * apply fourier transform on the matrix

* train():
    * convertToGaussian(x)
    * K ij = κ(x i , x j )
    * alpha = dft(y) ./ (dft(k) + lambda

* update():
    * x = getFeatures(frame)
    * K ij = κ(x i , x j )
    * train(x) 
    * return max(alpha*dft(k))


## Doubts:
* will have to lookup ridge regression
* gaussian space
* how to get the output mapped to the frames

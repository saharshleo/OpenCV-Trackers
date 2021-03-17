### Introduction:

The Aim is to implement the Online Boosting Tracker using the Adaptive Boosting Algorithm.

### Known disadvantages:
* execution time dependent on size of search region or object being tracked
* don't know how to decide hyperparameters - number of rows, features, selectors
* no way of deciding if the object has left the frame / is occluded

### Algo:

![algo](../assets/boosting_flowchart.png)

**main**
* video_path, opencv cap, videoWriter, ask for roi
* #rows = `N`
* #col / #features = `F`
* #selectors = `S`
* #weak_clf_to_replace = `R`
* initialize Boosting object (`bo`)
* `bo`.build_features
* `bo`.train_weak_clf
* `bo`.init_sample_weights
* `bo`.init_selector_pool
* `bo`.get_strong_clf
* 
* LOOP till q pressed or video finished:
  * read frame
  * get search region
  * `bo`.get_confidence_map
  * `bo`.get_bbox
  * `bo`.update_strong_clf
  * 

**class Boosting**
* __init__(frame, roi, N, F, S, R) ==> init blue rect

* build_features() ==> 
  * LOOP 1 to F:
    * choose random feature (Haar / LBP / Histogram)
    * choose random type (if Haar then specify kernel type and scale)
    * feature_list = {f_type(Haar): { kernel_type: , scale: , meu+: , meu-: , theta: , polarity: , alpha: , error: , lambda_wrong: , lambda_corr: , selector_index: }, ......} (think if dict is better or list)

* train_weak_clf() ==> 
  * LOOP from 1 to N:
    * choose random location from blue rect
    * compute all features
    * X[i] = computed features, y[i] = according to location +1/-1
  * LOOP on all features:
    * compute meu+, meu-
    * theta, polarity

* init_sample_weights() ==> 
  * assign 1/N to all sample (X[i])

* init_selector_pool() ==>
  * LOOP 1 to S:
    * choose random (previously unselected) feature from feature_list
  
* get_strong_clf() ==> 
  * LOOP 1 to S:
    * compute error of all features of that selector
    * select min error feature
    * calculate alpha of that feature
    * append to strong clf
    * update weights of sample (lambda)

* get_confidence_map() ==> 
  * eval strong clf on every pixel of blue rect

* get_bbox() ==> 
  * mean_shift on confidence_map

* update_strong_clf() ==>
  * LOOP 1 to R:
    * get max error feature from strong clf
    * compute all features of that selector's pool on new blue rect
    * compute meu+, meu-, theta, polarity
    * compute error
    * select min error feature and add to strong clf


### Reference: 
1. https://www.researchgate.net/publication/221259753_Real-Time_Tracking_via_On-line_Boosting
2. https://ieeexplore.ieee.org/document/5459285
3. https://ieeexplore.ieee.org/abstract/document/1640768

### TO DO:
* Parallelize/improve the method used to find the confidence map
* Implement the method to update strong classifier

### Doubts / Things to try:

* random location samples or fix location by specifying white and black region coords
* update strong classifier logic (only replace or get min error feature for all selector)
* meu+, meu- from kalman filter or not

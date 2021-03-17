### ALGO

**main**
* videopath, opencv cap, roi
* intitalize CSRT class object (tracker)
* tracker.generate_features()
* tracker.get_spatial_reliability_map()
* tracker.get_constrainted_filter()
* tracker.get_weights()
* LOOP till q pressed or video finished:
    * read frame
    * new roi = get_new_position()
    * tracker.update()

**class CSRT**
* Generate_features()
    * generate multi-channel HOG/colorname filters

* get_spatial_reliability_map()
    * p(y|m=1,x) (appearance likelihood)
        * bayes rules of color histogram ==> cf, cb
    * prior p(m=1)
        * ratio of region size of foreground/background histogram extraction
    * weak spatial prior p(x|m=1)
        * Epanechnikov kernel

* get_constrainted_filter()
    * LOOP OVER channels:
        * use regression and map(m) and return filter h

* get_weights()
    * LOOP over channels:
        * learning reliablity
            * wl = convolute f*h based on error
        * detection reliablity
            * ratio of two major bumps in response map
            * wd = 1-min(max2/max1,0.5)
        * w = wl*wd

* get_new_position():

* update()
    * extract and update foreground and background histogram
    * estimate reliability map m
    * estimate new filter h using m
    * estimate channel reliablity w from h
    * update filter
    * update channel reliablity

### Doubts
* spaitial reliablity map not clear for implementation
* multichannel filtes means rgb channels or something else
* do we need to run filter over whole image or on some patch
* formulas for regression not clear


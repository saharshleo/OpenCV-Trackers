### Algorithm

**def Lucas_Kanade_Tracker(polarity, video, pixel):**

    polarity = 1 -> forward tracking trajectory
    
    polarity = -1 -> Backward tracking trajectory
    
    #####################
    
    tracking code
    
    #####################
    
    return trajectory # trajectory is a list containing trajectory points

**def calc_FB_error_of_all_points(points_to_track, video):**
    
    # Many methods exist for this, easiest one will be the following one
    
    error = {}
    
    for point in points_to_track:
    
    trajectory1 = Lucas_Kanade_Tracker(1, video, point)
    
    trajectory2 = Lucas_Kanade_Tracker(-1, video, point)
    
    error['point'] = abs(trajectory1[-1] - trajectory2[-1])
    
    return error

**def chose_pixels_to_track(pixel_array, threshold):**
    
    # pixel array hashes pixel's individual FB errors with their identity
    
    points_to_track = []
    
    # Adding pixel co-ordinates for tracking happens in this function
    
    for pixel in pixel_array:
    
    if(pixel[1] < threshold):
    
    points_to_track.append([pixel[0][0], pixel[0][1]) 
    
    return points_to_track

**def form_bounding_box(point_positions):**
    
    <!--
    
    Estimation of the bounding box displacement from
    
    the remaining points is performed using median over
    
    each spatial dimension. Scale change is computed as
    
    follows: for each pair of points, a ratio between current
    
    point distance and previous point distance is computed;
    
    bounding box scale change is defined as the median
    
    over these ratios. An implicit assumption of the point-
    
    based representation is that the object is composed of
    
    small rigid patches. Parts of the objects that do not sat-
    
    isfy this assumption (object boundary, flexible parts) are
    
    not considered in the voting since they are rejected by
    
    the error measure.
    
    -->

**def create_initial_set_of_tracking_points(roi):**
    
    initial_tracking_points = []
    
    # Write code to fill above list
    
    return initial_tracking_points

**def main():**
    
    for each frame
    
    - load video
    
    - draw roi
    
    - create_initial_set_of_tracking_points
    
    - cacl_FB_error_of_all_points
    
    - choose_pixels_to_track
    
    - form_bounding_box



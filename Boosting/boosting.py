import cv2
import numpy as np
import random
from skimage.transform import integral_image
import multiprocessing
from multiprocessing import Pool

# Helper class
from feature_haar import FeatureHaar, compute_haar_feature

class Boosting:
    def __init__(self, frame, object_roi, num_rows, num_features, num_selectors, num_to_replace):
        self.frame = frame
        self.object_roi = object_roi
        self.num_rows = num_rows
        self.num_features = num_features
        self.num_selectors = num_selectors
        self.num_to_replace = num_to_replace

        self.get_blue_roi()

        # self.blue_roi
        # self.training_data, # self.num_pos_samples, # self.num_neg_samples
        # self.training_rows, # self.training_labels
        # self.feature_list, # self.feature_info
        # self.weights
        # self.selector_pool
        self.strong_classifier = {}
    
    def get_blue_roi(self):
        tl_x = max(0, int(self.object_roi[0] - self.object_roi[2]/2))
        tl_y = max(0, int(self.object_roi[1] - self.object_roi[3]/2))
        
        # TODO: should be in range of image
        w = 2*self.object_roi[2]
        h = 2*self.object_roi[3]

        self.blue_roi = [tl_x, tl_y, w, h]

        # Debugging
        # c = self.frame.copy()
        # cv2.rectangle(c, (tl_x, tl_y), (tl_x+w, tl_y+h), (0, 255, 0), 2)
        # cv2.imshow("frame", c)
        # key = cv2.waitKey(0)
        # if key & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()

    
    def build_features(self):
        self.feature_list = []
        self.feature_info = {}
        
        for feature_id in range(self.num_features):
            f = FeatureHaar()

            f.generateRandomFeature(self.object_roi[3], self.object_roi[2])

            self.feature_list.append(f)
            self.feature_info[feature_id] = {
                'meu+': 0,
                'meu-': 0,
                'theta': 0,
                'polarity': 0,
                'alpha': 0,
                'error': 0,
                'lambda_wrong': 0,
                'lambda_correct': 0,
                'selector_index': -1
            }            


    def get_training_data(self):
        # TODO: should be in range of image, with equal height and width as that of image
        t_x, t_y = self.blue_roi[0], self.blue_roi[1]
        c_x, c_y = self.blue_roi[0] + self.blue_roi[2], self.blue_roi[1] + self.blue_roi[3]
        b_x, b_y = self.blue_roi[0] + 2*self.blue_roi[2], self.blue_roi[1] + 2*self.blue_roi[3]
        
        pos_image = integral_image(self.frame[(t_y+c_y)//2:(b_y+c_y)//2, (t_x+c_x)//2:(b_x+c_x)//2])
        
        self.training_data = [
            (integral_image(self.frame[t_y:c_y, t_x:c_x]), -1),
            (pos_image, 1),
            (integral_image(self.frame[t_y:c_y, c_x:b_x]), -1),
            (pos_image, 1),
            (integral_image(self.frame[c_y:b_y, t_x:c_x]), -1),
            (pos_image, 1),
            (integral_image(self.frame[c_y:b_y, c_x:b_x]), -1),
            (pos_image, 1)
        ]
        
        self.num_pos_samples = 4
        self.num_neg_samples = 4


    def get_weak_classifiers(self):
        self.get_training_data()

        self.weights = self.init_sample_weights(len(self.training_data), self.num_pos_samples, self.num_neg_samples)
        
        # Apply features in training data
        self.training_labels = np.array(list(map(lambda data: data[1], self.training_data)))
        self.training_rows = np.zeros((len(self.training_data), len(self.feature_list)))
        i = 0
        for img, label in self.training_data:
            self.training_rows[i] = (list(map(lambda f: compute_haar_feature(img, f.featureType, f.location), self.feature_list)))
            i += 1

        for feature_id in range(len(self.feature_list)):
            feature_values = self.training_rows[:, feature_id]

            pos_sum, neg_sum = 0, 0
            for f_value_id in range(len(feature_values)):
                if self.training_labels[f_value_id] == -1:
                    neg_sum += self.weights[f_value_id]

                elif self.training_labels[f_value_id] == 1:
                    pos_sum += self.weights[f_value_id]

            meu_plus = pos_sum / self.num_pos_samples
            meu_minus = neg_sum / self.num_neg_samples

            self.feature_info[feature_id]['meu+'] = meu_plus
            self.feature_info[feature_id]['meu-'] = meu_minus

            self.feature_info[feature_id]['theta'] = abs(meu_plus + meu_minus) / 2
            self.feature_info[feature_id]['polarity'] = 1 if (meu_plus - meu_minus)>=0 else -1


    def init_sample_weights(self, len_training_data, num_pos, num_neg):
        return np.ones(len_training_data)


    def init_selector_pool(self):
        feature_id_list = list(range(self.num_features))

        random.shuffle(feature_id_list)

        f_per_s = self.num_features // self.num_selectors
        self.selector_pool = {}
        for sel_id in range(self.num_selectors):
            self.selector_pool[sel_id] = feature_id_list[sel_id*f_per_s:sel_id*f_per_s+f_per_s]

        # print(self.selector_pool)


    def get_strong_classifier(self):

        for sel_id in self.selector_pool:
            self.weights = self.weights / np.linalg.norm(self.weights)

            # Compute error of each feature and select min error feature
            min_feature_id = -1
            min_feature_pred_labels = []
            for feature_id in self.selector_pool[sel_id]:
                h_x = lambda f_x: self.feature_info[feature_id]['polarity'] * (1 if (f_x - self.feature_info[feature_id]['theta'])>=0 else -1)
                predicted_labels = list(map(lambda f_x: h_x(f_x), self.training_rows[:, feature_id]))

                lambda_wrong = 0
                lambda_correct = 0
                
                for sample_id in range(len(predicted_labels)):
                    if self.training_labels[sample_id] == predicted_labels[sample_id]:
                        lambda_correct += self.weights[sample_id]

                    else:
                        lambda_wrong += self.weights[sample_id]

                self.feature_info[feature_id]['error'] = lambda_wrong / (lambda_wrong + lambda_correct)

                # Min error feature from that pool
                if min_feature_id == -1:
                    min_feature_id = feature_id
                    min_feature_pred_labels = predicted_labels
                elif self.feature_info[feature_id]['error'] < self.feature_info[min_feature_id]['error']:
                    min_feature_id = feature_id
                    min_feature_pred_labels = predicted_labels


            self.strong_classifier[sel_id] = {'feature_id': None, 'alpha': None}
            self.strong_classifier[sel_id]['feature_id'] = min_feature_id

            min_error = self.feature_info[min_feature_id]['error']
            if min_error == 0:
                self.strong_classifier[sel_id]['alpha'] = 0
                continue

            self.strong_classifier[sel_id]['alpha'] = (1/2)*np.log((1-min_error)/min_error)

            # Update sample weights
            for sample_id in range(len(self.training_labels)):
                if min_feature_pred_labels[sample_id] == self.training_labels[sample_id]:
                    self.weights[sample_id] = self.weights[sample_id] * (1/ (2 * (1 - min_error)))
                else:
                    self.weights[sample_id] = self.weights[sample_id] * (1/ (2 * min_error))


    '''
    def get_confidence_map(self):
        self.confidence_map = np.zeros((self.blue_roi[2], self.blue_roi[3]), dtype=np.float)

        # TODO: applying on all pixels (misclassified if feature not applicable)
        # TODO: if using window then check range

        w, h = self.blue_roi[2], self.blue_roi[3]
        for x in range(self.blue_roi[0], self.blue_roi[0]+self.blue_roi[2]):

            for y in range(self.blue_roi[1], self.blue_roi[1]+self.blue_roi[3]):
                tl_x = max(0, int(self.object_roi[0] - self.object_roi[2]/2))
                tl_y = max(0, int(self.object_roi[1] - self.object_roi[3]/2))
                br_x = min(tl_x+self.blue_roi[2]//2, self.frame.shape[1])
                br_y = min(tl_y+self.blue_roi[3]//2, self.frame.shape[0])
                
                image_integral = integral_image(self.frame[tl_y:br_y, tl_x:br_x])

                conf = 0

                for sel_id in self.strong_classifier:
                    f_id = self.strong_classifier[sel_id]['feature_id']
                    f = self.feature_list[f_id]
                    
                    f_x = compute_haar_feature(image_integral, f.featureType, f.location)
                    h_x = self.feature_info[f_id]['polarity'] * (1 if (f_x - self.feature_info[f_id]['theta'])>=0 else -1)

                    conf += (h_x * self.strong_classifier[sel_id]['alpha'])

                self.confidence_map[x-self.blue_roi[0]][y-self.blue_roi[1]] = conf
    '''

    def get_confidence_map(self):
        w, h = self.blue_roi[2], self.blue_roi[3]

        self.confidence_map = np.zeros((w, h), dtype=np.float)

        print("[DEBUG] Initial Confidence map: {}".format(self.confidence_map))

        pool = Pool()
        results = []
        count = 0
        print("Total = {}".format(w*h))
        for x in range(self.blue_roi[0], self.blue_roi[0]+self.blue_roi[2]):

            for y in range(self.blue_roi[1], self.blue_roi[1]+self.blue_roi[3]):
                tl_x = max(0, int(x - self.object_roi[2]//2))
                tl_y = max(0, int(y - self.object_roi[3]//2))
                br_x = min(x+self.object_roi[2]//2, self.frame.shape[1])
                br_y = min(y+self.object_roi[3]//2, self.frame.shape[0])

                # r = pool.apply_async(self.parallel_helper, [tl_x, tl_y, br_x, br_y, x, y, count])

                # r = multiprocessing.Process(target=self.parallel_helper, args=(tl_x, tl_y, br_x, br_y, x, y, count,))
                # results.append(r)

                # r.start()

                self.parallel_helper(tl_x, tl_y, br_x, br_y, x, y, count)

                count += 1

        # for process in results:
        #     process.join()
        
        # [result.get() for result in results]

        print("[DEBUG] Final Confidence map: {}".format(self.confidence_map))



    def parallel_helper(self, tl_x, tl_y, br_x, br_y, x, y, count):
        image_integral = integral_image(self.frame[tl_y:br_y, tl_x:br_x])

        conf = 0
        # print("Started_{}!".format(count))
        for sel_id in self.strong_classifier:
            f_id = self.strong_classifier[sel_id]['feature_id']
            f = self.feature_list[f_id]
            
            f_x = compute_haar_feature(image_integral, f.featureType, f.location)
            h_x = self.feature_info[f_id]['polarity'] * (1 if (f_x - self.feature_info[f_id]['theta'])>=0 else -1)
            # print(f_x, self.feature_info[f_id]['theta'], self.feature_info[f_id]['polarity'], h_x, self.strong_classifier[sel_id]['alpha'])

            conf += (h_x * self.strong_classifier[sel_id]['alpha'])

        # print()
        # print("End_{}!".format(count))
        self.confidence_map[x-self.blue_roi[0]][y-self.blue_roi[1]] = conf


    def get_bbox(self):
        track_window = self.object_roi    # col, row, w, h

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(self.confidence_map, track_window, term_crit)

        # self.object_roi = track_window

        x, y, w, h = self.object_roi

        self.object_roi = [1, 2, 3, 4]

        self.object_roi[0] = track_window[0] + x
        self.object_roi[1] = track_window[1] + y
        self.object_roi[2] = w
        self.object_roi[3] = h

        self.object_roi = tuple(self.object_roi)


    def update_strong_classifier(self):
        # Get max error feature
        max_error = -float('inf')
        max_feature_id = -1
        max_sel_id = -1
        
        for sel_id in self.strong_classifier:
            f_id = self.strong_classifier[sel_id]['feature_id']
            if max_error < self.feature_info[f_id]['error']:
                max_error = self.feature_info[f_id]['error']
                max_feature_id = f_id
                max_sel_id = sel_id

        
        self.get_blue_roi()

        # ######## From get_weak_classifiers()
        self.get_training_data()

        self.weights = self.init_sample_weights(len(self.training_data), self.num_pos_samples, self.num_neg_samples)
        
        # Apply features in training data
        self.training_labels = np.array(list(map(lambda data: data[1], self.training_data)))
        self.training_rows = np.zeros((len(self.training_data), len(self.selector_pool[max_sel_id])))
        i = 0
        for img, label in self.training_data:
            # self.training_rows[i] = (list(map(lambda f: compute_haar_feature(img, f.featureType, f.location), self.feature_list)))
            j = 0
            for f_id in self.selector_pool[max_sel_id]:
                f = self.feature_list[f_id]
                self.training_rows[i][j] = compute_haar_feature(img, f.featureType, f.location)
                j += 1

            i += 1

        i = 0
        for feature_id in self.selector_pool[max_sel_id]:
            feature_values = self.training_rows[:, i]

            pos_sum, neg_sum = 0, 0
            for f_value_id in range(len(feature_values)):
                if self.training_labels[f_value_id] == -1:
                    neg_sum += self.weights[f_value_id]

                elif self.training_labels[f_value_id] == 1:
                    pos_sum += self.weights[f_value_id]

            meu_plus = pos_sum / self.num_pos_samples
            meu_minus = neg_sum / self.num_neg_samples

            self.feature_info[feature_id]['meu+'] = meu_plus
            self.feature_info[feature_id]['meu-'] = meu_minus

            self.feature_info[feature_id]['theta'] = abs(meu_plus + meu_minus) / 2
            self.feature_info[feature_id]['polarity'] = 1 if (meu_plus - meu_minus)>=0 else -1

        # ######## From get_weak_classifiers()

        # ######## From get_strong_classifiers()
        self.weights = self.weights / np.linalg.norm(self.weights)

        # Compute error of each feature and select min error feature
        min_feature_id = -1
        min_feature_pred_labels = []
        i = 0
        for feature_id in self.selector_pool[max_sel_id]:
            h_x = lambda f_x: self.feature_info[feature_id]['polarity'] * (1 if (f_x - self.feature_info[feature_id]['theta'])>=0 else -1)
            predicted_labels = list(map(lambda f_x: h_x(f_x), self.training_rows[:, i]))

            lambda_wrong = 0
            lambda_correct = 0
            
            for sample_id in range(len(predicted_labels)):
                if self.training_labels[sample_id] == predicted_labels[sample_id]:
                    lambda_correct += self.weights[sample_id]

                else:
                    lambda_wrong += self.weights[sample_id]

            self.feature_info[feature_id]['error'] = lambda_wrong / (lambda_wrong + lambda_correct)

            # Min error feature from that pool
            if min_feature_id == -1:
                min_feature_id = feature_id
                min_feature_pred_labels = predicted_labels
            elif self.feature_info[feature_id]['error'] < self.feature_info[min_feature_id]['error']:
                min_feature_id = feature_id
                min_feature_pred_labels = predicted_labels

            i += 1


        self.strong_classifier[sel_id] = {'feature_id': None, 'alpha': None}
        self.strong_classifier[sel_id]['feature_id'] = min_feature_id

        min_error = self.feature_info[min_feature_id]['error']
        if min_error == 0:
            self.strong_classifier[sel_id]['alpha'] = 0
            # continue

        self.strong_classifier[sel_id]['alpha'] = (1/2)*np.log((1-min_error)/min_error)

        # ######## From get_strong_classifiers()
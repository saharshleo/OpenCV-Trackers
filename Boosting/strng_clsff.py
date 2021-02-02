# Initial common_importance weight for all weak classifiers is 1(lambda).
# I assume we will be having a 2 dimensional array/list storing features of all weak classifiers.
# We will have a separate two dimensional array/list for storing all corresponding lambda values.
# We will have a separate one dimensional array/list for storing weights of weak classifiers(alpha).

# n->no. of weak classifiers,m->no. of features in a weak classifier
# strong_classifier->two dimensional list containing all weak classifiers and their corresponding
# features, weight_pos -> corresponds to (lambda)(corr)[n][m], weight_neg -> corresponds to (lamda)(wrong)[n][m]
# expected_values -> 2 dimensional list containing all expected values of features in the strong classifier
# common_importance_weight -> corresponds to lamda, error -> two dimensional list which will be used
# for storing errors of all features of weak classifiers present in strong classifier
# classifier error -> corresponds to en, classifier_weight -> corresponds to alpha
# feature_pool -> Dictionary containing global pool of features. 

import numpy as np
import random

common_importance_weight = 1
def strong_classifier_update(n, m, strong_classifier, weight_pos, weight_neg,expected_values,
                             common_importance_weight,error,classifier_error,classifier_weight,
                             feature_pool):
    for i in range(n):
        for j in range(m):
            if(strong_classifier[i][j] == expected_values[i][j]):
                weight_pos[i][j] += common_importance_weight
            else:
                weight_neg[i][j] += common_importance_weight
            error[i][j] = weight_neg[i][j]/(weight_neg[i][j] + weight_pos[i][j])

        # Now,take the best feature which has minimum error and assign it as error of whole weak 
        # weak classifier.
        m_best = error[i].index(min(error[i]))
        classifier_error[i] = error[i][m_best]

        # Take precautions such that denominator in log term must not become zero
        if(classifier_error[i] == 0 or classifier_error[i] > 0.5):
            return

        # Update classifier weight on the basis of classifier error
        classifier_weight = 0.5*np.log((1-classifier_error[i])/classifier_error[i])
        
        # We may require a threshold variable for this comparison
        if(strong_classifier[i][m_best] == expected_values[i][m_best]):
            common_importance_weight = common_importance_weight*0.5/(1-classifier_error[i])
        else:
            common_importance_weight = common_importance_weight*0.5/classifier_error[i]

        worst_feature_index = error[i].index(max(error[i]))

        # Replace worst weak classifier in the pool with a new one
        one_feature_selector(strong_classifier[i],worst_feature_index,feature_pool)


def one_feature_selector(weak_classifier, index, feature_pool):
    num = feature_pool[random.randrange(len(feature_pool))]['selector']
    while(num != -1):
        num = feature_pool[random.randrange(len(feature_pool))]['selector']
    weak_classifier[index] = feature_pool[num]['value']
    # Update expected_values list accordingly during training


def get_strong_classifier(n, m, strong_classifier, feature_pool):
    for i in range(n):
        for j in range(m):
            num1 = random.randrange(len(feature_pool))
            if(feature_pool[num1]['selector'] == -1):
                strong_classifier[i][j] = feature_pool[num1]['value']
            else:
                j -= 1
# Referred algorithm is : 
# Algorithm 2.1 On-line AdaBoost for feature selection
# Require: training example x, y, y ∈ {−1, +1}
# Require: strong classifier h strong (initialized randomly)
# wrong
# Require: weights λ corr
# (initialized with 1)
# n,m , λ n,m
# initialize the importance weight λ = 1
# // for all selectors
# for n = 1, 2, .., N do
# // update the selector h sel
# n
# for m = 1, 2, .., M do
# // update each weak classifier
# weak
# h weak
# n,m = update(h n,m , x, y, λ)
# // estimate errors
# if h weak
# n,m (x) = y then
# corr
# λ corr
# n,m = λ n,m + λ
# else
# λ wrong
# = λ wrong
# + λ
# n,m
# n,m
# end if
# λ wrong
# e n,m = λ corr n,m
# wrong
# n,m +λ n,m
# end for
# // choose weak classifier with the lowest error
# m + = arg min m (e n,m )
# weak
# e n = e n,m + ; h sel
# n = h n,m +
# 1
# if e n = 0 or e n > 2 then
# exit
# end if
# // calculate voting
# weight
# 
# 
# 1−e n
# 1
# α n = 2 · ln e n
# // update importance weight
# if h sel
# n (x) = y then
# 1
# λ = λ · 2·(1−e
# n )
# else
# λ = λ · 2·e 1 n
# end if
# // replace worst weak classifier with a new one
# m − = arg max m (e n,m )
# wrong
# λ corr
# n,m − = 1; λ n,m − = 1;
# get new h weak
# n,m −
# end for
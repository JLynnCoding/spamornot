# -*- coding: utf-8 -*-
"""
Performs analysis of data from file. Imports data, gets means, standard
deviations, and heatmaps of features separated by class. Tests groupings of 
features on numerous classifiers. Tests the best two performing classifiers
on experimental data.
"""

import pandas as pd
import seaborn as sns
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt


def get_data_from_file(filename):
    """
    Retrieve data from CSV file.

    Parameters
    ----------
    None
    
    Returns
    -------
    insta_df : dataframe containing data retrieved from file with updated 
    labels.

    """
    # Import Instagram data from CSV                       
    data = pd.read_csv(filename)
    print("Data imported from " + filename)
    
    return data
    
def get_corr_and_heatmap(df, title, filename):
    """
    Compute correlation and plot heatmap for features of df.

    Parameters
    ----------
    df : Dataframe to be used to compute correlation and heatmap

    Returns
    -------
    None.

    """
    m = df.corr()
    fig, ax = plt.subplots(figsize=(10,10))    # Sample figsize in inches
    #color = plt.get_cmap('RdYlGn')   # 
    #color.set_bad('lightblue') 
    sns.heatmap(m, annot = True, linewidths = .5, ax = ax)
    plt.title(title)
    plt.savefig(filename)
    print("\nHeatmap created and saved as " + filename)
    
    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim() # Determine the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    
    plt.show()

def get_mean_and_sd(df, feature):
    """
    Calculates the mean and standard deviation for df[feature] and returns 
    both as floats, rounded to two decimal places.

    Parameters
    ----------
    df : Dataframe of data to be used to make calculations.
    feature : String representing label of column for which mean and standard
    deviation must be calculated in df.

    Returns
    -------
    mn : Float representing the mean of the values in df[feature]
    sd : Float representing the standard deviation of the values in 
    df[feature]

    """
    values_to_check = np.array(df[feature])
    mn = round(np.mean(values_to_check), 2)
    sd = round(np.std(values_to_check), 2)
    return mn, sd

def get_mean_and_sd_table(df, features, filename):
    """
    Get mean and standard deviation for each column in features from df, and 
    save to a CSV. 

    Parameters
    ----------
    df : Dataframe of data to be used to make calculations.
    features : List containing labels of columns for which mean and standard
    deviation must be calculated in df.
    filename : String representing filename to be used to save CSV.

    Returns
    -------
    None.
    """
    
    # Create blank list with table headers as first line
    table_headers_mn_sd = ["Feature", "Mean", "Standard Deviation"]
    df_mn_sd = []
    df_mn_sd.append(table_headers_mn_sd)
    
    # Calculate means and standard deviations for all columns in features
    for each in features:
        mn, sd = get_mean_and_sd(df, each)
        value_list = [each, mn, sd]
        df_mn_sd.append(value_list)
    
    # Save as CSV
    with open(filename + '.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        for row in df_mn_sd:
            csvwriter.writerow(row)   
    
    print("\nThe table has been saved as " + filename + ".csv")
        
def get_pair_plot(df, features, label, value, filename):    
    """
    Plots pairwise relationships for list of features for label == value for 
    the random train sample and saves it as a jpg.

    Parameters
    ----------
    df : Dataframe of data to be used to make calculations.
    features : List containing labels of columns for which mean and standard
    deviation must be calculated in df.
    label : String representing column label to be used for conditional.
    value : Variable containing value to be checked in conditional.
    filename : String representing filename to be used to save plot.

    Returns
    -------
    None.
    """
    # Split dataset 50/50 into train and testing parts
    train_df,test_df = train_test_split(df,test_size=0.5)
    
    # Plot pairwise relationships for each class and save each to pdf
    sns.pairplot(train_df[train_df[label] == value][features])
    plt.savefig(filename + ".jpg")  
    plt.show()                                                
                                                       
    print("The pair plot has been saved as " + filename +  ".jpg")

def get_knn_predictions(train_df, test_df, features, label, k_list):
    """
    Scales x data and trains k-nn classifier on train_df, then predicts labels
    for test_df. Returns the dataframe with an added column with the 
    predictions for each k value.

    Parameters
    ----------
    train_df : Dataframe split for training.
    test_df: Dataframe split for testing.
    features : List of features to be considered in training.
    label : String representing column name for y values.
    k_list : List of integers that are values for k.

    Returns
    -------
    test_df : dataframe with added columns for k predictions.
    """
    
    # Train and test for each k-NN
    for each in k_list:
        x = np.array(train_df[features].values)
        y = np.array(train_df[label].values)

        # Scale data
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        knn_classifier = KNeighborsClassifier(n_neighbors=each)
        knn_classifier.fit(x, np.ravel(y))
        
        test_df = test_df.copy() # Copy created to avoid working with slice
    
        # Add predictions in new column for k-NN
        test_df['k ' + str(each)] = \
            knn_classifier.predict(test_df[features].values)
            
    return test_df

def get_knn_accuracies(df, k, label):
    """
    Checks the accuracy of the k-nn predictions and returns a list of 
    k values with their accuracies as a percent rounded to two decimal places.
    Determines the highest accuracy and returns the highest accuracy as well 
    as the corresponding k value.

    Parameters
    ----------
    df : Dataframe with values to be checked for accuracy.
    label : String representing column name for true values to be checked.
    k : List of k values to be used to find appropriate columns
    label : String representing column name for y values.

    Returns
    -------
    knn_accuracies : List of k values and floats for their respective 
    accuracies as a percent, rounded to two decimal places.
    max_k : Integer representing the k value that produced the highest 
    prediction accuracy.
    max_acc : Float representing the highest prediction accuracy, which was 
    produced with k = max_k.
    """
    knn_accuracies = []
    max_acc = 0
    max_k = 0
    
    for each in k:
        accuracy = accuracy_score(df[label], 
                                  df['k ' + str(each)])
        value_list = [each, round((accuracy * 100), 2)]
        knn_accuracies.append(value_list)
        if accuracy > max_acc:
            max_k = each
            max_acc = round((accuracy * 100), 2)
         
    return knn_accuracies, max_k, max_acc

def get_logistic_regression_predictions(train_df, test_df, features, label):
    """
    Trains and tests predictions on provided dataframes using logistic 
    prediction and the features listed in the features list that is passed to
    the function.

    Parameters
    ----------
    train_df : Dataframe for training logistic regression.
    test_df : Dataframe for testing logistic regression's predictions
    features : List of features to be considered.
    label : String representing column for y values.

    Returns
    -------
    test_df : Dataframe ammended with 'log_regres' column containing class 
    predictions from the logistic regression.

    """
    x = train_df[features].values
    y = train_df[label].values

    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    log_regr_classifier = LogisticRegression()
    log_regr_classifier.fit(x, np.ravel(y))
    
    test_df = test_df.copy()
    test_df['log regres']\
        = log_regr_classifier.predict(test_df[features].values)
        
    return test_df        

def get_naive_bayesian_predict(train_df, test_df, features, label):
    """
    Trains NB on train_df, predicts labels, and adds predictions in a new
    "NB" column.

    Parameters
    ----------
    train_df : Dataframe with data split for training.
    test_df : Dataframe with data split for training.
    features : List of strings representing features to use for x values for
    training/predictions.
    label : String representing column to use for y values for 
    training/predictions.

    Returns
    -------
    test_df : Dataframe with data split for testing.
    """
    # Get x and y values to train NB
    x = np.array(train_df[features].values)
    y = np.array(train_df[label].values)

    NB_classifier = GaussianNB().fit(x, y) # Train NB
    
    test_df = test_df.copy() # Copy to not work with a view
    
    # Predict labels and add in "NB" column
    test_df['NB']\
        = NB_classifier.predict(test_df[features].values)
        
    return test_df

def get_decision_tree_predict(train_df, test_df, features, label):
    """
    Trains Decision Tree Classifier using train_df and predicts label values 
    for test_df. Returns test_df with added "Dec Tree" column with predicted
    values.

    Parameters
    ----------
    train_df : Dataframe with data split for training.
    test_df : Dataframe with data split for training.
    features : List of strings representing features to use for x values for
    training/predictions.
    label : String representing column to use for y values for 
    training/predictions.

    Returns
    -------
    test_df : Dataframe with data split for testing.
    """
    # Get x and y values to train Decision Tree
    x = np.array(train_df[features].values)
    y = np.array(train_df[label].values)

    dec_tree_classifier = tree.DecisionTreeClassifier(criterion = 'entropy')
    dec_tree_classifier = dec_tree_classifier.fit(x,y) # Train Decision Tree
    
    test_df = test_df.copy() # Copy to not work with a view
    
    # Predict labels and add in "Dec Tree" column
    test_df['Dec Tree']\
        = dec_tree_classifier.predict(test_df[features].values)
        
    return test_df

def get_random_forest_predict(train_df, test_df, features, label, n, d):
    """
    Trains Random Forest Classifier using n_estimator of n, and max_depth of 
    d, and train_df to predicts label values for test_df. Returns test_df with 
    added "Rnd For NnDd" column with predicted values.

    Parameters
    ----------
    train_df : Dataframe with data split for training.
    test_df : Dataframe with data split for training.
    features : List of strings representing features to use for x values for
    training/predictions.
    label : String representing column to use for y values for 
    training/predictions.
    n : Integer representing n_estimators
    d : Integer representing max_depth

    Returns
    -------
    test_df : Dataframe with data split for testing.
    """
    # Get x and y values to train Random Forest 
    x = np.array(train_df[features].values)
    y = np.array(train_df[label].values)

    rnd_frst_classifier = RandomForestClassifier(n_estimators = n, 
                                                 max_depth = d, 
                                                 criterion = 'entropy')
    rnd_frst_classifier = rnd_frst_classifier.fit(x,y) # Train Random Forest
    
    test_df = test_df.copy() # Copy to not work with a view
    
    # Predict labels and add in "Rnd For NnDd" column
    test_df['Rnd For N' + str(n) + 'D' + str(d)]\
        = rnd_frst_classifier.predict(test_df[features].values)
        
    return test_df

def get_error_rate(df, label, n, d):
    """
    Calculates the error rate for the predictions of Random Forest with n 
    n_estimators and d max_depth and returns it as a float.

    Parameters
    ----------
    df : Dataframe to be checked for predictions against the true values.
    label : Label for the true values.
    n : integer representing n_estimators used in Random Forest.
    d : integer representing max_depth used in Random Forest.

    Returns
    -------
    error_rate : float that represents the error rate for predictions by
    Random Forest with n n_estimators and d max_depth. 

    """
    error_rate = np.mean(df['Rnd For N' + str(n) + 'D' + str(d)] != 
                           df[label])
    return error_rate  

def check_accuracy(df, label):
    """
    Checks the accuracy of the predictions in the column corresponding to 
    label and returns four integers representing true and false 1s and true 
    and false 0s.

    Parameters
    ----------
    df : Dataframe containing label and true values to be checked.
    label : String representing column of predictions to be checked for
    accuracy.

    Returns
    -------
    correct_norm_predictions : Integer representing number of true positive 
    predictions.
    incorrect_pos_predictions :  Integer representing number of false positive 
    predictions.
    correct_neg_predictions :  Integer representing number of true negative 
    predictions.
    incorrect_neg_predictions :  Integer representing number of false negative 
    predictions.
    """
    # Initialize counts
    correct_pos_predictions = 0
    incorrect_pos_predictions = 0
    correct_neg_predictions = 0
    incorrect_neg_predictions = 0

    # Iterate through df checking 'fake' against the value in label column
    for i in df.index:
       if df[label][i] == 1:
           if df['fake'][i] == 1:
               correct_pos_predictions += 1
           else: 
               incorrect_pos_predictions += 1
       if df[label][i] == 0:
           if df['fake'][i] == 0:
               correct_neg_predictions += 1
           else: 
               incorrect_neg_predictions += 1
               
    return correct_pos_predictions, incorrect_pos_predictions, \
        correct_neg_predictions, incorrect_neg_predictions

def get_accuracy_or_prob_percentage(count_for_numerator,\
                                          count_for_total_for_denominator):
    """
    Calculates the accuracy or probability and returns it as a percentage,
    rounded to two decimal places.
    
    Parameters
    ----------
    count_for_probability : a number that represents the occurrence of 
    whatever the accuracy or probability is being evaluated for.
    count_for_total: a number that represents the alternative outcomes against
    which the accuracy or probability is being calculated.

    Returns
    -------
    Returns the probability as a percentage, rounded to 2 decimal places.  
    """
    return round((count_for_numerator / (count_for_total_for_denominator + \
                                  count_for_numerator) * 100), 2)

def get_simple_prediction(df):
    '''
    Applies classifier to dataset and returns list of predictions for the 
    'class' value in order.

    Parameters
    ----------
    df : Dataframe to be checked against classifier.

    Returns
    -------
    prediction_list : List of 'class' values derived from classifier applied
    to each row in df. 
    '''
    prediction_list = []
    for i in df.index: 
        if (df['full=user'][i] < 0.25) or (df['URL'][i] != 0) and \
            ((df['posts'][i] > 50) and (df['pic'][i] == 1)):
            prediction_list.append(0)
        else:
            prediction_list.append(1)
    return prediction_list

def main():
      
    # Uncomment to view entire dataframe
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.max_rows', None)
    #pd.set_option('display.width', None)
    #pd.set_option('display.max_colwidth', None)
    
    # Get data from file
    insta_df = get_data_from_file("datasets_145755_339960_instagram.csv")
    
    
    # Simplify labels
    insta_df = insta_df.rename(columns={'profile pic': 'pic',
                                    'nums/length username': 'user #:len', 
                                    'nums/length fullname': 'full #:len',
                                    'fullname words': 'full words',
                                    'name==username': 'full=user',
                                    'description length': 'desc len',
                                    'external URL': 'URL',
                                    '#posts': 'posts',
                                    '#followers': 'followers',
                                    '#follows': 'follows'})
    
    # List of the features for reference 
    features = ['pic', 'user #:len', 'full words', 'full #:len', 
                'full=user', 'desc len', 'URL', 'private', 'posts', 
                'followers', 'follows']
       
    # Reduced list of features as determined from visual inspection of means
    # and deviation table
    reduced_features_1 = ['pic', 'user #:len', 'full words', 'desc len', 
                          'URL', 'posts', 'followers', 'follows']
    
    # Reduced list of features as determined from suspected significant
    # features based on means and standard deviations
    reduced_features_2 = ['pic','desc len', 'URL', 'posts', 'followers', 
                          'follows']
    
    # Reduced list of features suspected to be significant but disregarding
    # booleans
    reduced_features_3 = ['desc len', 'posts', 'followers', 
                          'follows']
    
    # Reduced list of features suspected to be significant in classifying
    # real or spam accounts
    reduced_features_4 = ['posts', 'followers', 'follows']
    
    # Reduced list of features suspected to be significant in classifying
    # real or spam accounts
    reduced_features_5 = ['followers', 'follows']
    
    features_list = [features, reduced_features_1, reduced_features_2, 
                 reduced_features_3, reduced_features_4, reduced_features_5]
        
    # Create separate dataframes for genuine ("fake" == 0) or fake 
    # ("fake" == 1) data
    insta_df_0 = insta_df[insta_df["fake"] == 0][features].copy()
    insta_df_1 = insta_df[insta_df["fake"] == 1][features].copy()

    # Compute mean and standard deviation for all features for insta_df_0,
    # insta_df_1, and insta_df and save to CSVs
    get_mean_and_sd_table(insta_df_0, features, 'insta_df_0_mn_sd')
    get_mean_and_sd_table(insta_df_1, features, 'insta_df_1_mn_sd')
    get_mean_and_sd_table(insta_df, features, 'insta_df_mn_sd')
    
    insta_df_0_red = insta_df[insta_df["fake"] == \
                              0][reduced_features_1].copy()
    insta_df_1_red = insta_df[insta_df["fake"] == \
                              1][reduced_features_1].copy()
    
    # Compute correlation and plot heatmaps for each dataframe
    get_corr_and_heatmap(insta_df_0_red, 
                         "Heatmap for Genuine Instagram Accounts " + 
                         "('fake' == 0)","insta_0_heatmap_red.jpg")
    get_corr_and_heatmap(insta_df_1_red, 
                         "Heatmap for Fake Instagram Accounts " + 
                         "('fake' == 1)","insta_1_heatmap_red.jpg")
    
    """ # Commenting out because not useful
    # Plot pairwise relationships for each class and save each to pdf
    get_pair_plot(insta_df, reduced_features, 'fake', 0, 'insta_0_pairwise')                         
    get_pair_plot(insta_df, reduced_features, 'fake', 1, 'insta_1_pairwise')                         
    """            
    
    # K-NN Predictions
    k = [3, 5, 7, 9, 11]
    label = 'fake'
    
    # Split dataset 50/50 into train and testing parts
    train_df, test_df = train_test_split(insta_df, test_size=0.5)
    
    # K-NN with All Features
    
    knn_predict_test_df = get_knn_predictions(train_df, test_df, features, 
                                              label, k)
    
    value_list, best_k, max_acc = get_knn_accuracies(knn_predict_test_df, 
                                                        k, label)
    
    print("\nBest k is " + str(best_k) + " with an accuracy of " + 
          str(max_acc) + "% when considering all features.")
    
    knn__tn_all, knn_fp_all, knn_fn_all, knn_tp_all = \
            confusion_matrix(knn_predict_test_df[label], 
                             knn_predict_test_df['k ' + str(best_k)]).ravel()
    
    print("\nThe confusion matrix for k = " + str(best_k) + ":")
    print("[[" + str( knn__tn_all) + " " + str(knn_fp_all) + "]\n [" + 
          str(knn_fn_all) + " " + str(knn_tp_all) + "]]") 
    
    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(knn_tp_all, knn_fn_all)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(knn__tn_all, knn_fp_all)) + "%")
    
    # K-NN with Reduced_1 Features
    knn_predict_test_df_red = \
        get_knn_predictions(train_df, test_df, reduced_features_1, label, k)
    
    value_list_red, best_k_red, max_acc_red = \
        get_knn_accuracies(knn_predict_test_df_red, k, label)

    print("\nBest k is " + str(best_k_red) + " with an accuracy of " + 
          str(max_acc_red) + "% when considering only 'pic', 'user #:len'," + 
          "'full words', 'full=user', 'desc len', 'URL','posts', " + 
          "'followers', 'follows'.")
    
    knn__tn_red1, knn_fp_red1, knn_fn_red1, knn_tp_red1 = \
            confusion_matrix(knn_predict_test_df_red[label], 
                             knn_predict_test_df_red['k ' + 
                                                 str(best_k_red)]).ravel()
    
    print("\nThe confusion matrix for k = " + str(best_k_red) + ":")
    print("[[" + str( knn__tn_red1) + " " + str(knn_fp_red1) + "]\n [" + 
          str(knn_fn_red1) + " " + str(knn_tp_red1) + "]]") 
    
    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(knn_tp_red1, knn_fn_red1)) + 
          "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(knn__tn_red1, knn_fp_red1)) + 
          "%")
    
    
    # K-NN with Reduced_2 Features
    knn_predict_test_df_red_2 = \
        get_knn_predictions(train_df, test_df, reduced_features_2, label, k)
    
    value_list_red_2, best_k_red_2, max_acc_red_2 = \
        get_knn_accuracies(knn_predict_test_df_red_2, k, label)

    print("\nBest k is " + str(best_k_red_2) + " with an accuracy of " + 
          str(max_acc_red_2) + "% when considering only 'pic','desc len'," +
          "'URL', 'posts', 'followers', 'follows'.")
    
    knn__tn_red2, knn_fp_red2, knn_fn_red2, knn_tp_red2 = \
            confusion_matrix(knn_predict_test_df_red_2[label], 
                             knn_predict_test_df_red_2['k ' + 
                                                 str(best_k_red_2)]).ravel()
    
    print("\nThe confusion matrix for k = " + str(best_k_red_2) + ":")
    print("[[" + str( knn__tn_red2) + " " + str(knn_fp_red2) + "]\n [" + 
          str(knn_fn_red2) + " " + str(knn_tp_red2) + "]]") 
    
    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(knn_tp_red2, knn_fn_red2)) + 
          "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(knn__tn_red2, knn_fp_red2)) + 
          "%")
    
    
    # K-NN with Reduced_3 Features
    knn_predict_test_df_red_3 = \
        get_knn_predictions(train_df, test_df, reduced_features_3, label, k)
    
    value_list_red_3, best_k_red_3, max_acc_red_3 = \
        get_knn_accuracies(knn_predict_test_df_red_3, k, label)

    print("\nBest k is " + str(best_k_red_3) + " with an accuracy of " + 
          str(max_acc_red_3) + "% when considering only 'desc len', 'posts'" +
          ", 'followers', 'follows'.")
    
    knn__tn_red3, knn_fp_red3, knn_fn_red3, knn_tp_red3 = \
            confusion_matrix(knn_predict_test_df_red_3[label], 
                             knn_predict_test_df_red_3['k ' + 
                                                 str(best_k_red_3)]).ravel()
    
    print("\nThe confusion matrix for k = " + str(best_k_red_3) + ":")
    print("[[" + str( knn__tn_red3) + " " + str(knn_fp_red3) + "]\n [" + 
          str(knn_fn_red3) + " " + str(knn_tp_red3) + "]]") 
    
    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(knn_tp_red3, knn_fn_red3)) + 
          "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(knn__tn_red3, knn_fp_red3)) + 
          "%")
    
    # K-NN with Reduced_4 Features
    knn_predict_test_df_red4 = \
        get_knn_predictions(train_df, test_df, reduced_features_4, label, k)
    
    value_list_red_4, best_k_red_4, max_acc_red_4 = \
        get_knn_accuracies(knn_predict_test_df_red4, k, label)

    print("\nBest k is " + str(best_k_red_4) + " with an accuracy of " + 
          str(max_acc_red_4) + "% when considering only 'posts', " +
          "'followers', 'follows'.")
    
    knn__tn_red4, knn_fp_red4, knn_fn_red4, knn_tp_red4 = \
            confusion_matrix(knn_predict_test_df_red4[label], 
                             knn_predict_test_df_red4['k ' + 
                                                 str(best_k_red_4)]).ravel()
    
    print("\nThe confusion matrix for k = " + str(best_k_red_4) + ":")
    print("[[" + str( knn__tn_red4) + " " + str(knn_fp_red4) + "]\n [" + 
          str(knn_fn_red4) + " " + str(knn_tp_red4) + "]]") 
    
    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(knn_tp_red4, knn_fn_red4)) + 
          "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(knn__tn_red4, knn_fp_red4)) + 
          "%")
    
    # K-NN with Reduced_5 Features
    knn_predict_test_df_red5 = \
        get_knn_predictions(train_df, test_df, reduced_features_5, label, k)
    
    value_list_red_5, best_k_red_5, max_acc_red_5 = \
        get_knn_accuracies(knn_predict_test_df_red5, k, label)

    print("\nBest k is " + str(best_k_red_5) + " with an accuracy of " + 
          str(max_acc_red_5) + "% when considering only 'followers', " +
          "'follows'.")
    
    knn__tn_red5, knn_fp_red5, knn_fn_red5, knn_tp_red5 = \
            confusion_matrix(knn_predict_test_df_red5[label], 
                             knn_predict_test_df_red5['k ' + 
                                                 str(best_k_red_5)]).ravel()
    
    print("\nThe confusion matrix for k = " + str(best_k_red_5) + ":")
    print("[[" + str( knn__tn_red5) + " " + str(knn_fp_red5) + "]\n [" + 
          str(knn_fn_red5) + " " + str(knn_tp_red5) + "]]") 
    
    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(knn_tp_red5, knn_fn_red5)) + 
          "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(knn__tn_red5, knn_fp_red5)) + 
          "%")
    
    # Logistic Regression
    # Split dataset 50/50 into train and testing parts
    train_df, test_df = train_test_split(insta_df, test_size=0.5)
    
    # Logistic Regression with All Features
    
    logreg_predict_test_all_df = \
        get_logistic_regression_predictions(train_df, test_df, features, 
                                            label)
        
    logreg_all_acc = \
        round((accuracy_score(logreg_predict_test_all_df[label],
                                   logreg_predict_test_all_df['log regres'])
                     * 100), 2)
        
    print("\nLogistic regression with all features has an accuracy of " + 
          str(logreg_all_acc) + "%.")
    
    logreg_tn_all, logreg_fp_all, logreg_fn_all, logreg_tp_all = \
            confusion_matrix(logreg_predict_test_all_df[label], 
                             logreg_predict_test_all_df['log regres']).ravel()
    
    print("\nThe confusion matrix for logistic regression with all features" +
          "considered: ")
    print("[[" + str(logreg_tn_all) + " " + str(logreg_fp_all) + "]\n [" + 
          str(logreg_fn_all) + " " + str(logreg_tp_all) + "]]")     
    
    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(logreg_tp_all, logreg_fn_all)) + 
          "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(logreg_tn_all, logreg_fp_all)) + 
          "%")    
        
    # Logistic Regression with Reduced_1 Features
    logreg_predict_test_red1_df = \
        get_logistic_regression_predictions(train_df, test_df, 
                                            reduced_features_1, label)
        
    logreg_red1_acc = \
        round((accuracy_score(logreg_predict_test_red1_df[label],
                                   logreg_predict_test_red1_df['log regres']) 
                     * 100), 2)
        
    print("\nLogistic regression considering only 'pic', 'user #:len'," + 
          "'full words', 'full=user', 'desc len', 'URL','posts', " + 
          "'followers', 'follows' has an accuracy of " + 
          str(logreg_red1_acc)+ "%.")
    
    logreg_tn_red1, logreg_fp_red1, logreg_fn_red1, logreg_tp_red1 = \
            confusion_matrix(logreg_predict_test_red1_df[label], 
                             logreg_predict_test_red1_df['log regres']).ravel()
    
    print("\nThe confusion matrix for logistic regression with reduced " +
          "features 1 considered: ")
    
    print("[[" + str(logreg_tn_red1) + " " + str(logreg_fp_red1) + "]\n [" + 
          str(logreg_fn_red1) + " " + str(logreg_tp_red1) + "]]")    
    
    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(logreg_tp_red1, 
                                              logreg_fn_red1)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(logreg_tn_red1, 
                                              logreg_fp_red1)) + "%")    
    
    # Logistic Regression with Reduced_2 Features
    
    logreg_predict_test_red2_df = \
        get_logistic_regression_predictions(train_df, test_df, 
                                            reduced_features_2, label)
        
    logreg_red2_acc = \
        round((accuracy_score(logreg_predict_test_red2_df[label],
                                   logreg_predict_test_red2_df['log regres']) 
                     * 100), 2)
        
    print("\nLogistic regression considering only 'pic','desc len'," +
          "'URL', 'posts', 'followers', 'follows' has an accuracy of " + 
          str(logreg_red2_acc)+ "%.")
    
    logreg_tn_red2, logreg_fp_red2, logreg_fn_red2, logreg_tp_red2 = \
            confusion_matrix(logreg_predict_test_red2_df[label], 
                             logreg_predict_test_red2_df\
                                 ['log regres']).ravel()
    
    print("\nThe confusion matrix for logistic regression with reduced " +
          "features 2 considered: ")
    
    print("[[" + str(logreg_tn_red2) + " " + str(logreg_fp_red2) + "]\n [" + 
          str(logreg_fn_red2) + " " + str(logreg_tp_red2) + "]]")  
    
    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(logreg_tp_red2, 
                                              logreg_fn_red2)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(logreg_tn_red2, 
                                              logreg_fp_red2)) + "%")    
    
    # Logistic Regression with Reduced_3 Features
    
    logreg_predict_test_red3_df = \
        get_logistic_regression_predictions(train_df, test_df, 
                                            reduced_features_3, label)
        
    logreg_red3_acc = \
        round((accuracy_score(logreg_predict_test_red3_df[label],
                                   logreg_predict_test_red3_df['log regres']) 
                     * 100), 2)
        
    print("\nLogistic regression considering only ''desc len', 'posts', " +
          "'followers', 'follows' has an accuracy of " + 
          str(logreg_red3_acc)+ "%.")
    
    logreg_tn_red3, logreg_fp_red3, logreg_fn_red3, logreg_tp_red3 = \
            confusion_matrix(logreg_predict_test_red3_df[label], 
                             logreg_predict_test_red3_df\
                                 ['log regres']).ravel()
    
    print("\nThe confusion matrix for logistic regression with reduced " +
          "features 3 considered: ")
    
    print("[[" + str(logreg_tn_red3) + " " + str(logreg_fp_red3) + "]\n [" + 
          str(logreg_fn_red3) + " " + str(logreg_tp_red3) + "]]")  

    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(logreg_tp_red3, 
                                              logreg_fn_red3)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(logreg_tn_red3, 
                                              logreg_fp_red3)) + "%")       
    
    # Logistic Regression with Reduced_4 Features
    logreg_predict_test_red4_df = \
        get_logistic_regression_predictions(train_df, test_df, 
                                            reduced_features_4, label)
        
    logreg_red4_acc = \
        round((accuracy_score(logreg_predict_test_red4_df[label],
                                   logreg_predict_test_red4_df['log regres']) 
                     * 100), 2)
        
    print("\nLogistic regression considering only 'followers', 'follows' " + 
          "has an accuracy of " + str(logreg_red4_acc)+ "%.")
    
    logreg_tn_red4, logreg_fp_red4, logreg_fn_red4, logreg_tp_red4 = \
            confusion_matrix(logreg_predict_test_red4_df[label], 
                             logreg_predict_test_red4_df\
                                 ['log regres']).ravel()
    
    print("\nThe confusion matrix for logistic regression with reduced " +
          "features 4 considered: ")
    
    print("[[" + str(logreg_tn_red4) + " " + str(logreg_fp_red4) + "]\n [" + 
          str(logreg_fn_red4) + " " + str(logreg_tp_red4) + "]]")     
    
    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(logreg_tp_red4, 
                                              logreg_fn_red4)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(logreg_tn_red4, 
                                              logreg_fp_red4)) + "%")    

    # Logistic Regression with Reduced_5 Features
    
    logreg_predict_test_red5_df = \
        get_logistic_regression_predictions(train_df, test_df, 
                                            reduced_features_5, label)
        
    logreg_red5_acc = \
        round((accuracy_score(logreg_predict_test_red5_df[label],
                                   logreg_predict_test_red5_df['log regres']) 
                     * 100), 2)
        
    print("\nLogistic regression considering only 'followers','follows' has" +
          " an accuracy of " + str(logreg_red5_acc)+ "%.")
    
    logreg_tn_red5, logreg_fp_red5, logreg_fn_red5, logreg_tp_red5 = \
            confusion_matrix(logreg_predict_test_red5_df[label], 
                             logreg_predict_test_red5_df\
                                 ['log regres']).ravel()
    
    print("\nThe confusion matrix for logistic regression with reduced " +
          "features 5 considered: ")
    
    print("[[" + str(logreg_tn_red5) + " " + str(logreg_fp_red5) + "]\n [" + 
          str(logreg_fn_red5) + " " + str(logreg_tp_red5) + "]]")  

    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(logreg_tp_red5, 
                                              logreg_fn_red5)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(logreg_tn_red5, 
                                              logreg_fp_red5)) + "%")    

    
    # Naive Bayesian (NB)
    
    # Split dataset 50/50 into train and testing parts
    train_df, test_df = train_test_split(insta_df, test_size=0.5)
    
    # NB with All Features    
    nb_predict_test_all_df = \
        get_naive_bayesian_predict(train_df, test_df, features, label)

    nb_all_acc = round((accuracy_score(nb_predict_test_all_df[label],
                                    nb_predict_test_all_df['NB'])* 100), 2)

    print("\nNaive Bayesian with all features has an accuracy of " + 
          str(nb_all_acc) + "%.")
    
    nb_tn_all, nb_fp_all, nb_fn_all, nb_tp_all = \
            confusion_matrix(nb_predict_test_all_df[label], 
                             nb_predict_test_all_df['NB']).ravel()
    
    print("\nThe confusion matrix for Naive Bayesian with all features " +
          "considered: ")
    
    print("[[" + str(nb_tn_all) + " " + str(nb_fp_all) + "]\n [" + 
          str(nb_fn_all) + " " + str(nb_tp_all) + "]]")      
    
    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(nb_tp_all, 
                                              nb_fn_all)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(nb_tn_all, 
                                              nb_fp_all)) + "%")   
    
    # NB with Reduced Features 1    
    nb_predict_test_red1_df = \
        get_naive_bayesian_predict(train_df, test_df, reduced_features_1, 
                                   label)
    nb_red1_acc = round((accuracy_score(nb_predict_test_red1_df[label],
                                    nb_predict_test_red1_df['NB'])* 100), 2)

    print("\nNaive Bayesian considering only 'pic', 'user #:len'," + 
          "'full words', 'full=user', 'desc len', 'URL','posts', " + 
          "'followers', 'follows' has an accuracy of " + 
          str(nb_red1_acc) + "%.")
    
    nb_tn_red1, nb_fp_red1, nb_fn_red1, nb_tp_red1 = \
            confusion_matrix(nb_predict_test_red1_df[label], 
                             nb_predict_test_red1_df['NB']).ravel()
    
    print("\nThe confusion matrix for Naive Bayesian with Reduced Features " +
          "1 considered: ")
    
    print("[[" + str(nb_tn_red1) + " " + str(nb_fp_red1) + "]\n [" + 
          str(nb_fn_red1) + " " + str(nb_tp_red1) + "]]")

    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(nb_tp_red1, 
                                              nb_fn_red1)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(nb_tn_red1, 
                                              nb_fp_red1)) + "%")   
    
    # NB with Reduced Features 2    
    nb_predict_test_red2_df = \
        get_naive_bayesian_predict(train_df, test_df, reduced_features_2, 
                                   label)
        
    nb_red2_acc = round((accuracy_score(nb_predict_test_red2_df[label],
                                    nb_predict_test_red2_df['NB'])* 100), 2)

    print("\nNaive Bayesian considering only 'pic','desc len'," +
          "'URL', 'posts', 'followers', 'follows' has an accuracy of " + 
          str(nb_red2_acc) + "%.")
    
    nb_tn_red2, nb_fp_red2, nb_fn_red2, nb_tp_red2 = \
            confusion_matrix(nb_predict_test_red2_df[label], 
                             nb_predict_test_red2_df['NB']).ravel()
    
    print("\nThe confusion matrix for Naive Bayesian with Reduced Features " +
          "2 considered: ")
    
    print("[[" + str(nb_tn_red2) + " " + str(nb_fp_red2) + "]\n [" + 
          str(nb_fn_red2) + " " + str(nb_tp_red2) + "]]")

    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(nb_tp_red2, 
                                              nb_fn_red2)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(nb_tn_red2, 
                                              nb_fp_red2)) + "%")   
    
    
    # NB with Reduced Features 3   
    nb_predict_test_red3_df = \
        get_naive_bayesian_predict(train_df, test_df, reduced_features_3, 
                                   label)
        
    nb_red3_acc = round((accuracy_score(nb_predict_test_red3_df[label],
                                    nb_predict_test_red3_df['NB'])* 100), 2)

    print("\nNaive Bayesian considering only 'desc len', 'posts', " +
          "'followers', 'follows'' has an accuracy of " + str(nb_red3_acc) + 
          "%.")    
    
    nb_tn_red3, nb_fp_red3, nb_fn_red3, nb_tp_red3 = \
            confusion_matrix(nb_predict_test_red3_df[label], 
                             nb_predict_test_red3_df['NB']).ravel()
    
    print("\nThe confusion matrix for Naive Bayesian with Reduced Features " +
          "3 considered: ")
    
    print("[[" + str(nb_tn_red3) + " " + str(nb_fp_red3) + "]\n [" + 
          str(nb_fn_red3) + " " + str(nb_tp_red3) + "]]")

    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(nb_tp_red3, 
                                              nb_fn_red3)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(nb_tn_red3, 
                                              nb_fp_red3)) + "%")   
    
        
    # NB with Reduced Features 4   
    nb_predict_test_red4_df = \
        get_naive_bayesian_predict(train_df, test_df, reduced_features_4, 
                                   label)
        
    nb_red4_acc = round((accuracy_score(nb_predict_test_red4_df[label],
                                    nb_predict_test_red4_df['NB'])* 100), 2)

    print("\nNaive Bayesian considering only 'posts', " +
          "'followers', 'follows'' has an accuracy of " + str(nb_red4_acc) + 
          "%.")    
    
    nb_tn_red4, nb_fp_red4, nb_fn_red4, nb_tp_red4 = \
            confusion_matrix(nb_predict_test_red4_df[label], 
                             nb_predict_test_red4_df['NB']).ravel()
    
    print("\nThe confusion matrix for Naive Bayesian with Reduced Features " +
          "4 considered: ")
    
    print("[[" + str(nb_tn_red4) + " " + str(nb_fp_red4) + "]\n [" + 
          str(nb_fn_red4) + " " + str(nb_tp_red4) + "]]")

    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(nb_tp_red4, 
                                              nb_fn_red4)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(nb_tn_red4, 
                                              nb_fp_red4)) + "%")   
    
    # NB with Reduced Features 5   
    nb_predict_test_red5_df = \
        get_naive_bayesian_predict(train_df, test_df, reduced_features_5, 
                                   label)
        
    nb_red5_acc = round((accuracy_score(nb_predict_test_red5_df[label],
                                    nb_predict_test_red5_df['NB'])* 100), 2)

    print("\nNaive Bayesian considering only 'followers', 'follows' " +
          " has an accuracy of " + str(nb_red5_acc) + "%.")    
    
    nb_tn_red5, nb_fp_red5, nb_fn_red5, nb_tp_red5 = \
            confusion_matrix(nb_predict_test_red5_df[label], 
                             nb_predict_test_red5_df['NB']).ravel()
    
    print("\nThe confusion matrix for Naive Bayesian with Reduced Features " +
          "5 considered: ")
    
    print("[[" + str(nb_tn_red5) + " " + str(nb_fp_red5) + "]\n [" + 
          str(nb_fn_red5) + " " + str(nb_tp_red5) + "]]")

    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(nb_tp_red5, 
                                              nb_fn_red5)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(nb_tn_red5, 
                                              nb_fp_red5)) + "%")   
    
    
    # Decision Tree
    
    # Split dataset 50/50 into train and testing parts
    train_df, test_df = train_test_split(insta_df, test_size=0.5)
    
    # Decision Tree with All Features    
    dectree_predict_test_all_df = \
        get_decision_tree_predict(train_df, test_df, features, label)

    dectree_all_acc = \
        round((accuracy_score(dectree_predict_test_all_df[label], 
                              dectree_predict_test_all_df['Dec Tree'])* 100), 
              2)

    print("\nDecision Tree with all features has an accuracy of " + 
          str(dectree_all_acc) + "%.")
    
    dectree_tn_all, dectree_fp_all, dectree_fn_all, dectree_tp_all = \
            confusion_matrix(dectree_predict_test_all_df[label], 
                             dectree_predict_test_all_df['Dec Tree']).ravel()
    
    print("\nThe confusion matrix for Decision Tree with all features " + 
          "considered: ")
    
    print("[[" + str(dectree_tn_all) + " " + str(dectree_fp_all) + "]\n [" + 
          str(dectree_fn_all) + " " + str(dectree_tp_all) + "]]")

    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(dectree_tp_all, 
                                              dectree_fn_all)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(dectree_tn_all, 
                                              dectree_fp_all)) + "%")     
    
    # Decision Tree with Reduced Features 1    
    dectree_predict_test_red1_df = \
        get_decision_tree_predict(train_df, test_df, reduced_features_1, 
                                  label)

    dectree_red1_acc = \
        round((accuracy_score(dectree_predict_test_red1_df[label], 
                              dectree_predict_test_red1_df['Dec Tree'])* 100), 
              2)

    print("\nDecision Tree considering only 'pic', 'user #:len'," + 
          "'full words', 'full=user', 'desc len', 'URL','posts', " + 
          "'followers', 'follows' has an accuracy of " + 
          str(dectree_red1_acc) + "%.")
    
    dectree_tn_red1, dectree_fp_red1, dectree_fn_red1, dectree_tp_red1 = \
            confusion_matrix(dectree_predict_test_red1_df[label], 
                             dectree_predict_test_red1_df['Dec Tree']).ravel()
    
    print("\nThe confusion matrix for Decision Tree with Reduced features 1" + 
          " considered: ")
    
    print("[[" + str(dectree_tn_red1) + " " + str(dectree_fp_red1) + "]\n [" + 
          str(dectree_fn_red1) + " " + str(dectree_tp_red1) + "]]")

    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(dectree_tp_red1, 
                                              dectree_fn_red1)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(dectree_tn_red1, 
                                              dectree_fp_red1)) + "%")    
    
    # Decision Tree with Reduced Features 2    
    dectree_predict_test_red2_df = \
        get_decision_tree_predict(train_df, test_df, reduced_features_2, 
                                  label)

    dectree_red2_acc = \
        round((accuracy_score(dectree_predict_test_red2_df[label], 
                              dectree_predict_test_red2_df['Dec Tree'])* 100), 
              2)

    print("\nDecision Tree considering only 'pic','desc len'," +
          "'URL', 'posts', 'followers', 'follows' has an accuracy of " + 
          str(dectree_red2_acc) + "%.")      
    
    dectree_tn_red2, dectree_fp_red2, dectree_fn_red2, dectree_tp_red2 = \
            confusion_matrix(dectree_predict_test_red2_df[label], 
                             dectree_predict_test_red2_df['Dec Tree']).ravel()
    
    print("\nThe confusion matrix for Decision Tree with Reduced features 2" + 
          " considered: ")
    
    print("[[" + str(dectree_tn_red2) + " " + str(dectree_fp_red2) + "]\n [" + 
          str(dectree_fn_red2) + " " + str(dectree_tp_red2) + "]]")    
    
    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(dectree_tp_red2, 
                                              dectree_fn_red2)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(dectree_tn_red2, 
                                              dectree_fp_red2)) + "%") 
    
    # Decision Tree with Reduced Features 3   
    dectree_predict_test_red3_df = \
        get_decision_tree_predict(train_df, test_df, reduced_features_3, 
                                  label)

    dectree_red3_acc = \
        round((accuracy_score(dectree_predict_test_red3_df[label], 
                              dectree_predict_test_red3_df['Dec Tree'])* 100), 
              2)

    print("\nDecision Tree considering only 'pic','desc len'," +
          "'URL', 'posts', 'followers', 'follows' has an accuracy of " + 
          str(dectree_red3_acc) + "%.")   

    dectree_tn_red3, dectree_fp_red3, dectree_fn_red3, dectree_tp_red3 = \
            confusion_matrix(dectree_predict_test_red3_df[label], 
                             dectree_predict_test_red3_df['Dec Tree']).ravel()
    
    print("\nThe confusion matrix for Decision Tree with Reduced features 3" + 
          " considered: ")
    
    print("[[" + str(dectree_tn_red3) + " " + str(dectree_fp_red3) + "]\n [" + 
          str(dectree_fn_red3) + " " + str(dectree_tp_red3) + "]]")  
    
    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(dectree_tp_red3, 
                                              dectree_fn_red3)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(dectree_tn_red3, 
                                              dectree_fp_red3)) + "%")     
        
    # Decision Tree with Reduced Features 4    
    dectree_predict_test_red4_df = \
        get_decision_tree_predict(train_df, test_df, reduced_features_4, 
                                  label)

    dectree_red4_acc = \
        round((accuracy_score(dectree_predict_test_red4_df[label], 
                              dectree_predict_test_red4_df['Dec Tree'])* 100), 
              2)

    print("\nDecision Tree considering only  'posts', 'followers', " +
          "'follows' has an accuracy of " + str(dectree_red4_acc) + "%.")      
    
    dectree_tn_red4, dectree_fp_red4, dectree_fn_red4, dectree_tp_red4 = \
            confusion_matrix(dectree_predict_test_red4_df[label], 
                             dectree_predict_test_red4_df['Dec Tree']).ravel()
    
    print("\nThe confusion matrix for Decision Tree with Reduced features 4" + 
          " considered: ")
    
    print("[[" + str(dectree_tn_red4) + " " + str(dectree_fp_red4) + "]\n [" + 
          str(dectree_fn_red4) + " " + str(dectree_tp_red4) + "]]")    

    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(dectree_tp_red4, 
                                              dectree_fn_red4)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(dectree_tn_red4, 
                                              dectree_fp_red4)) + "%")     
    
    # Decision Tree with Reduced Features 5   
    dectree_predict_test_red5_df = \
        get_decision_tree_predict(train_df, test_df, reduced_features_5, 
                                  label)

    dectree_red5_acc = \
        round((accuracy_score(dectree_predict_test_red5_df[label], 
                              dectree_predict_test_red5_df['Dec Tree'])* 100), 
              2)

    print("\nDecision Tree considering only 'followers', 'follows' " + 
          "has an accuracy of " + str(dectree_red5_acc) + "%.")   

    dectree_tn_red5, dectree_fp_red5, dectree_fn_red5, dectree_tp_red5 = \
            confusion_matrix(dectree_predict_test_red5_df[label], 
                             dectree_predict_test_red5_df['Dec Tree']).ravel()
    
    print("\nThe confusion matrix for Decision Tree with Reduced features 5" + 
          " considered: ")
    
    print("[[" + str(dectree_tn_red5) + " " + str(dectree_fp_red5) + "]\n [" + 
          str(dectree_fn_red5) + " " + str(dectree_tp_red5) + "]]")  
    
    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(dectree_tp_red5, 
                                              dectree_fn_red5)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(dectree_tn_red5, 
                                              dectree_fp_red5)) + "%")         
    
    # Random Forest
    
    # Split dataset 50/50 into train and testing parts
    train_df, test_df = train_test_split(insta_df, test_size=0.5)
    
    # Helper variables for tracking n and d values later
    n_d_list = [] # 2D List of all N and D values to be checked
    n_labels = [] # List of all N variables to serve as x labels in plot


    # Load lists
    for d in range(5):
        for n in range(10):
            n_d_list.append((n+1, d+1))
    for n in range (1, 11):
        n_labels.append(n)

    # Random Forest for All Features
    # Helper variables for tracking lowest error
    error_list_all = [] # List of error rates
    min_error_all = 100 # Min error set to high value to initalize
   
    # Loop through various N and D values to construct Random Forest 
    # Classifier, Predict XTest, and Computer Error Rates
    for each in n_d_list:
        
        # Predict values using Random Forest with n and d
        rndfor_all_test = get_random_forest_predict(
            train_df, test_df, features, label, 
            each[0], each[1])
        
        # Check Error rate for Random Forest with n and d
        current_error = round((get_error_rate(rndfor_all_test, label, each[0], 
                                         each[1]) * 100), 2)
        # Add Error rate to list of error rates
        error_list_all.append(current_error)
        
        # Check if this run has the smallest error rate and record, if so
        if current_error < min_error_all:
            min_error_all = current_error
            rndfor_best_test_all = rndfor_all_test
            
    #Plot error rates and the best combination of N and d.
    
    #error_list_all = np.array(error_list_all) # Converted for low overhead
    
    # Plot each d as separate line         
    d1 = plt.plot(n_labels, error_list_all[0:10], color='black', marker='o')
    d2 = plt.plot(n_labels, error_list_all[10:20], color='green', marker='o')
    d3 = plt.plot(n_labels, error_list_all[20:30], color='red', marker='o')
    d4 = plt.plot(n_labels, error_list_all[30:40], color='cyan', marker='o')
    d5 = plt.plot(n_labels, error_list_all[40:50], color='magenta', 
                  marker='o')
    plt.ylabel('Error Rate')
    plt.xlabel('Number of SubTrees (N)')
    plt.title('Random Forest Prediction Error Rates All Features')
    plt.legend((d1[0], d2[0], d3[0], d4[0], d5[0]), ('d = 1', 'd = 2', 
                                                     'd = 3', 'd = 4',
                                                     'd = 5'), 
                                                       title="Max Depths", 
                                                       fontsize=8)
    
    # Save plot to file
    plt.savefig('Random_Forest_Prediction_Error_Rates_All_Features.jpg')
    print("\nError rate plot saved as " +\
          "'Random_Forest_Prediction_Error_Rates_All_Features.jpg'")
    plt.show()  


    # Locating the lowest error in error_list to locate it in n_d_list
    lowest_error_all = error_list_all.index((min(error_list_all)))  # Convert to int

    # Create label for the best n and d for easy reference to column
    best_label_all = 'Rnd For N' + str(n_d_list[lowest_error_all][0]) +\
        'D' + str(n_d_list[lowest_error_all][1])

    # Find the best combination of n and d
    print("The best combination of n and d is for Random Forest with All" +
          " Features: " + str(n_d_list[lowest_error_all]) + " with an " +
          "error rate of " + str(min_error_all) + ".")

    # Calculate the accuracy for the best combination of N and d

    print("\nRandom Forest Predictions with Best N and d Combination " +
          "for All Features:")
    print(rndfor_best_test_all)

    # Check the accuracy of the predictions with the best N and d
    best_tp_all, best_fp_all, best_tn_all, best_fn_all = \
        check_accuracy(rndfor_best_test_all, best_label_all)

    # Get an accuracy percentage for this combination
    print("\nThe accuracy of n = " + str(n_d_list[lowest_error_all][0]) +
          " and d = " + str(n_d_list[lowest_error_all][1]) + " is " +
          str(get_accuracy_or_prob_percentage((best_tp_all + best_tn_all),
                                              (best_fp_all +
                                               best_fn_all))) + "%.")


    #  Compute the confusion matrix using the best combination of N and d

    rndfor_y_true_all = np.array(rndfor_best_test_all[label].values)
    rndfor_y_pred_all = \
        np.array(rndfor_best_test_all[best_label_all].values)
    rndfor_tn_all, rndfor_fp_all, rndfor_fn_all, rndfor_tp_all = \
        confusion_matrix(rndfor_y_true_all, rndfor_y_pred_all).ravel()

    print("\nThe confusion matrix for n = " +
          str(n_d_list[lowest_error_all][0]) + " and d  = " +
          str(n_d_list[lowest_error_all][1]) + ":")
    print("[[" + str(rndfor_tn_all) + " " + str(rndfor_fp_all) + "]\n [" +
          str(rndfor_fn_all) + " " + str(rndfor_tp_all) + "]]")

    print("TPR: " +
      str(get_accuracy_or_prob_percentage(rndfor_tp_all,
                                          rndfor_fn_all)) + "%")
    print("TNR: " +
          str(get_accuracy_or_prob_percentage(rndfor_tn_all,
                                              rndfor_fp_all)) + "%")


        
    # Random Forest for Reduced Features1
    # Helper variables for tracking lowest error
    error_list_red1 = [] # List of error rates
    min_error_red1 = 100 # Min error set to high value to initalize
   
    # Loop through various N and D values to construct Random Forest 
    # Classifier, Predict XTest, and Computer Error Rates
    for each in n_d_list:
        
        # Predict values using Random Forest with n and d
        rndfor_red1_test = get_random_forest_predict(
            train_df, test_df, reduced_features_1, label, 
            each[0], each[1])
        
        # Check Error rate for Random Forest with n and d
        current_error = round((get_error_rate(rndfor_red1_test, label, each[0], 
                                         each[1]) * 100), 2)
        
        # Add Error rate to list of error rates
        error_list_red1.append(current_error)
        
        # Check if this run has the smallest error rate and record, if so
        if current_error < min_error_red1:
            min_error_red1 = current_error
            rndfor_best_test_red1 = rndfor_red1_test
            
    #Plot error rates and the best combination of N and d.
    
    #error_list_red1 = np.array(error_list_red1) # Converted for low overhead
    
    # Plot each d as separate line         
    d1 = plt.plot(n_labels, error_list_red1[0:10], color='black', marker='o')
    d2 = plt.plot(n_labels, error_list_red1[10:20], color='green', marker='o')
    d3 = plt.plot(n_labels, error_list_red1[20:30], color='red', marker='o')
    d4 = plt.plot(n_labels, error_list_red1[30:40], color='cyan', marker='o')
    d5 = plt.plot(n_labels, error_list_red1[40:50], color='magenta', 
                  marker='o')
    plt.ylabel('Error Rate')
    plt.xlabel('Number of SubTrees (N)')
    plt.title('Random Forest Prediction Error Rates Reduced Features 1')
    plt.legend((d1[0], d2[0], d3[0], d4[0], d5[0]), ('d = 1', 'd = 2', 
                                                     'd = 3', 'd = 4',
                                                     'd = 5'), 
                                                       title="Max Depths", 
                                                       fontsize=8)
    
    # Save plot to file
    plt.savefig('Random_Forest_Prediction_Error_Rates_Red1_Features.jpg')
    print("\nError rate plot saved as " +\
          "'Random_Forest_Prediction_Error_Rates_Red1_Features.jpg'")
    plt.show()

    # Locating the lowest error in error_list to locate it in n_d_list
    lowest_error_red1 = error_list_red1.index((min(error_list_red1)))


    # Create label for the best n and d for easy reference to column
    best_label_red1 = ('Rnd For N' + str(n_d_list[lowest_error_red1][0]) +
    'D' + str(n_d_list[lowest_error_red1][1]))

    # Find the best combination of n and d
    print("The best combination of n and d is for Random Forest with " +
          "Reduced Features 1: " + str(n_d_list[lowest_error_red1]) +
          " with an error rate of " + str(min_error_red1) + "%")

    # Calculate the accuracy for the best combination of N and d
    print("\nRandom Forest Predictions with Best N and d Combination " +
          "for Reduced Features 1:")
    print(rndfor_best_test_red1)

    # Check the accuracy of the predictions with the best N and d
    best_tp_red1, best_fp_red1, best_tn_red1, best_fn_red1 = \
        check_accuracy(rndfor_best_test_red1, best_label_red1)

    # Get an accuracy percentage for this combination
    print("\nThe accuracy of n = " + str(n_d_list[lowest_error_red1][0]) +
          " and d = " + str(n_d_list[lowest_error_red1][1]) + " is " +
          str(get_accuracy_or_prob_percentage((best_tp_red1 +
                                               best_tn_red1),
                                              (best_fp_red1 +
                                               best_fn_red1))) + "%.")

    #  Compute the confusion matrix using the best combination of N and d

    rndfor_y_true_red1 = np.array(rndfor_best_test_red1[label].values)
    rndfor_y_pred_red1 = \
        np.array(rndfor_best_test_red1[best_label_red1].values)
    rndfor_tn_red1, rndfor_fp_red1, rndfor_fn_red1, rndfor_tp_red1 = \
        confusion_matrix(rndfor_y_true_red1, rndfor_y_pred_red1).ravel()

    print("\nThe confusion matrix for n = " +
          str(n_d_list[lowest_error_red1][0]) + " and d  = " +
          str(n_d_list[lowest_error_red1][1]) + ":")
    print("[[" + str(rndfor_tn_red1) + " " + str(rndfor_fp_red1) +
          "]\n [" + str(rndfor_fn_red1) + " " + str(rndfor_tp_red1) +
          "]]")

    print("TPR: " +
          str(get_accuracy_or_prob_percentage(rndfor_tp_red1,
                                          rndfor_fn_red1)) + "%")
    print("TNR: " +
          str(get_accuracy_or_prob_percentage(rndfor_tn_red1,
                                              rndfor_fp_red1)) + "%")

        
    # Random Forest for Reduced Features2
    # Helper variables for tracking lowest error
    error_list_red2 = [] # List of error rates
    min_error_red2 = 100 # Min error set to high value to initalize
   
    # Loop through various N and D values to construct Random Forest 
    # Classifier, Predict XTest, and Computer Error Rates
    for each in n_d_list:
        
        # Predict values using Random Forest with n and d
        rndfor_red2_test = get_random_forest_predict(
            train_df, test_df, reduced_features_2, label, 
            each[0], each[1])
        
        # Check Error rate for Random Forest with n and d
        current_error = round((get_error_rate(rndfor_red2_test, label, each[0], 
                                         each[1]) * 100), 2)
        # Add Error rate to list of error rates
        error_list_red2.append(current_error)
        
        # Check if this run has the smallest error rate and record, if so
        if current_error < min_error_red2:
            min_error_red2 = current_error
            rndfor_best_test_red2 = rndfor_red2_test
            
    #Plot error rates and the best combination of N and d.
    
    #error_list_red2 = np.array(error_list_red2) # Converted for low overhead
    
    # Plot each d as separate line         
    d1 = plt.plot(n_labels, error_list_red2[0:10], color='black', marker='o')
    d2 = plt.plot(n_labels, error_list_red2[10:20], color='green', marker='o')
    d3 = plt.plot(n_labels, error_list_red2[20:30], color='red', marker='o')
    d4 = plt.plot(n_labels, error_list_red2[30:40], color='cyan', marker='o')
    d5 = plt.plot(n_labels, error_list_red2[40:50], color='magenta', 
                  marker='o')
    plt.ylabel('Error Rate')
    plt.xlabel('Number of SubTrees (N)')
    plt.title('Random Forest Prediction Error Rates Reduced Features 2')
    plt.legend((d1[0], d2[0], d3[0], d4[0], d5[0]), ('d = 1', 'd = 2', 
                                                     'd = 3', 'd = 4',
                                                     'd = 5'), 
                                                       title="Max Depths", 
                                                       fontsize=8)
    
    # Save plot to file
    plt.savefig('Random_Forest_Prediction_Error_Rates_Red2_Features.jpg')
    print("\nError rate plot saved as " +\
          "'Random_Forest_Prediction_Error_Rates_Red2_Features.jpg'")
    plt.show()  


    # Locating the lowest error in error_list to locate it in n_d_list
    lowest_error_red2 = error_list_red2.index((min(error_list_red2)))

    # Create label for the best n and d for easy reference to column
    best_label_red2 = ('Rnd For N' + str(n_d_list[lowest_error_red2][0]) +
    'D' + str(n_d_list[lowest_error_red2][1]))

    # Find the best combination of n and d
    print("\nThe best combination of n and d is for Random Forest with " +
          " Reduced Features 2: " + str(n_d_list[lowest_error_red2]) +
          " with an error rate of " + str(min_error_red2) + "%")

    # Calculate the accuracy for the best combination of N and d
    print("\nRandom Forest Predictions with Best N and d Combination for"+
          " Reduced Features 2:")
    print(rndfor_best_test_red2)

    # Check the accuracy of the predictions with the best N and d
    best_tp_red2, best_fp_red2, best_tn_red2, best_fn_red2 = \
        check_accuracy(rndfor_best_test_red2, best_label_red2)

    # Get an accuracy percentage for this combination
    print("\nThe accuracy of n = " + str(n_d_list[lowest_error_red2][0]) +
          " and d = " + str(n_d_list[lowest_error_red2][1]) + " is " +
          str(get_accuracy_or_prob_percentage((best_tp_red2 +
                                               best_tn_red2),
                                              (best_fp_red2 +
                                               best_fn_red2))) + "%.")

    #  Compute the confusion matrix using the best combination of N and d

    rndfor_y_true_red2 = np.array(rndfor_best_test_red2[label].values)
    rndfor_y_pred_red2 = \
        np.array(rndfor_best_test_red2[best_label_red2].values)
    rndfor_tn_red2, rndfor_fp_red2, rndfor_fn_red2, rndfor_tp_red2 = \
        confusion_matrix(rndfor_y_true_red2, rndfor_y_pred_red2).ravel()

    print("\nThe confusion matrix for n = " +
          str(n_d_list[lowest_error_red2][0]) + " and d  = " +
          str(n_d_list[lowest_error_red2][1]) + ":")
    print("[[" + str(rndfor_tn_red2) + " " + str(rndfor_fp_red2) +
          "]\n [" + str(rndfor_fn_red2) + " " + str(rndfor_tp_red2) +
          "]]")

    print("TPR: " +
          str(get_accuracy_or_prob_percentage(rndfor_tp_red2,
                                          rndfor_fn_red2)) + "%")
    print("TNR: " +
          str(get_accuracy_or_prob_percentage(rndfor_tn_red2,
                                              rndfor_fp_red2)) + "%")

        
    # Random Forest for Reduced Features 3
    # Helper variables for tracking lowest error
    error_list_red3 = [] # List of error rates
    min_error_red3 = 100 # Min error set to high value to initalize
   
    # Loop through various N and D values to construct Random Forest 
    # Classifier, Predict XTest, and Computer Error Rates
    for each in n_d_list:
        
        # Predict values using Random Forest with n and d
        rndfor_red3_test = get_random_forest_predict(
            train_df, test_df, reduced_features_3, label, 
            each[0], each[1])
        
        # Check Error rate for Random Forest with n and d
        current_error = round((get_error_rate(rndfor_red3_test, label, each[0], 
                                         each[1]) * 100), 2)
        # Add Error rate to list of error rates
        error_list_red3.append(current_error)
        
        # Check if this run has the smallest error rate and record, if so
        if current_error < min_error_red3:
            min_error_red3 = current_error
            rndfor_best_test_red3 = rndfor_red3_test
            
    #Plot error rates and the best combination of N and d.
    
    #error_list_red2 = np.array(error_list_red2) # Converted for low overhead
    
    # Plot each d as separate line         
    d1 = plt.plot(n_labels, error_list_red3[0:10], color='black', marker='o')
    d2 = plt.plot(n_labels, error_list_red3[10:20], color='green', marker='o')
    d3 = plt.plot(n_labels, error_list_red3[20:30], color='red', marker='o')
    d4 = plt.plot(n_labels, error_list_red3[30:40], color='cyan', marker='o')
    d5 = plt.plot(n_labels, error_list_red3[40:50], color='magenta', 
                  marker='o')
    plt.ylabel('Error Rate')
    plt.xlabel('Number of SubTrees (N)')
    plt.title('Random Forest Prediction Error Rates Reduced Features 3')
    plt.legend((d1[0], d2[0], d3[0], d4[0], d5[0]), ('d = 1', 'd = 2', 
                                                     'd = 3', 'd = 4',
                                                     'd = 5'), 
                                                       title="Max Depths", 
                                                       fontsize=8)
    
    # Save plot to file
    plt.savefig('Random_Forest_Prediction_Error_Rates_Red3_Features.jpg')
    print("\nError rate plot saved as " +\
          "'Random_Forest_Prediction_Error_Rates_Red3_Features.jpg'")
    plt.show()  

    # Locating the lowest error in error_list to locate it in n_d_list
    lowest_error_red3 = error_list_red3.index((min(error_list_red3)))

    # Create label for the best n and d for easy reference to column
    best_label_red3 = ('Rnd For N' + str(n_d_list[lowest_error_red3][0]) +
    'D' + str(n_d_list[lowest_error_red3][1]))

    # Find the best combination of n and d
    print("\nThe best combination of n and d is for Random Forest with " +
          " Reduced Features 3: " + str(n_d_list[lowest_error_red3]) +
          " with an error rate of " + str(min_error_red3) + "%")

    # Calculate the accuracy for the best combination of N and d
    print("\nRandom Forest Predictions with Best N and d Combination for"+
          " Reduced Features 3:")
    print(rndfor_best_test_red3)

    # Check the accuracy of the predictions with the best N and d
    best_tp_red3, best_fp_red3, best_tn_red3, best_fn_red3 = \
        check_accuracy(rndfor_best_test_red3, best_label_red3)

    # Get an accuracy percentage for this combination
    print("\nThe accuracy of n = " + str(n_d_list[lowest_error_red3][0]) +
          " and d = " + str(n_d_list[lowest_error_red3][1]) + " is " +
          str(get_accuracy_or_prob_percentage((best_tp_red3 +
                                               best_tn_red3),
                                              (best_fp_red3 +
                                               best_fn_red3))) + "%.")

    #  Compute the confusion matrix using the best combination of N and d

    rndfor_y_true_red3 = np.array(rndfor_best_test_red3[label].values)
    rndfor_y_pred_red3 = \
        np.array(rndfor_best_test_red3[best_label_red3].values)
    rndfor_tn_red3, rndfor_fp_red3, rndfor_fn_red3, rndfor_tp_red3 = \
        confusion_matrix(rndfor_y_true_red3, rndfor_y_pred_red3).ravel()

    print("\nThe confusion matrix for n = " +
          str(n_d_list[lowest_error_red3][0]) + " and d  = " +
          str(n_d_list[lowest_error_red3][1]) + ":")
    print("[[" + str(rndfor_tn_red3) + " " + str(rndfor_fp_red3) +
          "]\n [" + str(rndfor_fn_red3) + " " + str(rndfor_tp_red3) +
          "]]")

    print("TPR: " +
          str(get_accuracy_or_prob_percentage(rndfor_tp_red3,
                                          rndfor_fn_red3)) + "%")
    print("TNR: " +
          str(get_accuracy_or_prob_percentage(rndfor_tn_red3,
                                              rndfor_fp_red3)) + "%")
        

    # Random Forest for Reduced Features 4
    # Helper variables for tracking lowest error
    error_list_red4 = [] # List of error rates
    min_error_red4 = 100 # Min error set to high value to initalize
   
    # Loop through various N and D values to construct Random Forest 
    # Classifier, Predict XTest, and Computer Error Rates
    for each in n_d_list:
        
        # Predict values using Random Forest with n and d
        rndfor_red4_test = get_random_forest_predict(
            train_df, test_df, reduced_features_4, label, 
            each[0], each[1])
        
        # Check Error rate for Random Forest with n and d
        current_error = round((get_error_rate(rndfor_red4_test, label, each[0], 
                                         each[1]) * 100), 2)
        # Add Error rate to list of error rates
        error_list_red4.append(current_error)
        
        # Check if this run has the smallest error rate and record, if so
        if current_error < min_error_red4:
            min_error_red4 = current_error
            rndfor_best_test_red4 = rndfor_red4_test
            
    #Plot error rates and the best combination of N and d.    
    # Plot each d as separate line         
    d1 = plt.plot(n_labels, error_list_red4[0:10], color='black', marker='o')
    d2 = plt.plot(n_labels, error_list_red4[10:20], color='green', marker='o')
    d3 = plt.plot(n_labels, error_list_red4[20:30], color='red', marker='o')
    d4 = plt.plot(n_labels, error_list_red4[30:40], color='cyan', marker='o')
    d5 = plt.plot(n_labels, error_list_red4[40:50], color='magenta', 
                  marker='o')
    plt.ylabel('Error Rate')
    plt.xlabel('Number of SubTrees (N)')
    plt.title('Random Forest Prediction Error Rates Reduced Features 4')
    plt.legend((d1[0], d2[0], d3[0], d4[0], d5[0]), ('d = 1', 'd = 2', 
                                                     'd = 3', 'd = 4',
                                                     'd = 5'), 
                                                       title="Max Depths", 
                                                       fontsize=8)
    
    # Save plot to file
    plt.savefig('Random_Forest_Prediction_Error_Rates_Red4_Features.jpg')
    print("\nError rate plot saved as " +\
          "'Random_Forest_Prediction_Error_Rates_Red4_Features.jpg'")
    plt.show()  


    # Locating the lowest error in error_list to locate it in n_d_list
    lowest_error_red4 = error_list_red4.index((min(error_list_red4)))

    # Create label for the best n and d for easy reference to column
    best_label_red4 = ('Rnd For N' + str(n_d_list[lowest_error_red4][0]) +
    'D' + str(n_d_list[lowest_error_red4][1]))

    # Find the best combination of n and d
    print("\nThe best combination of n and d is for Random Forest with " +
          " Reduced Features 4: " + str(n_d_list[lowest_error_red4]) +
          " with an error rate of " + str(min_error_red4) + "%")

    # Calculate the accuracy for the best combination of N and d
    print("\nRandom Forest Predictions with Best N and d Combination for"+
          " Reduced Features 4:")
    print(rndfor_best_test_red4)

    # Check the accuracy of the predictions with the best N and d
    best_tp_red4, best_fp_red4, best_tn_red4, best_fn_red4 = \
        check_accuracy(rndfor_best_test_red4, best_label_red4)

    # Get an accuracy percentage for this combination
    print("\nThe accuracy of n = " + str(n_d_list[lowest_error_red4][0]) +
          " and d = " + str(n_d_list[lowest_error_red4][1]) + " is " +
          str(get_accuracy_or_prob_percentage((best_tp_red4 +
                                               best_tn_red4),
                                              (best_fp_red4 +
                                               best_fn_red4))) + "%.")

    #  Compute the confusion matrix using the best combination of N and d

    rndfor_y_true_red4 = np.array(rndfor_best_test_red4[label].values)
    rndfor_y_pred_red4 = \
        np.array(rndfor_best_test_red4[best_label_red4].values)
    rndfor_tn_red4, rndfor_fp_red4, rndfor_fn_red4, rndfor_tp_red4 = \
        confusion_matrix(rndfor_y_true_red4, rndfor_y_pred_red4).ravel()

    print("\nThe confusion matrix for n = " +
          str(n_d_list[lowest_error_red4][0]) + " and d  = " +
          str(n_d_list[lowest_error_red4][1]) + ":")
    print("[[" + str(rndfor_tn_red4) + " " + str(rndfor_fp_red4) +
          "]\n [" + str(rndfor_fn_red4) + " " + str(rndfor_tp_red4) +
          "]]")

    print("TPR: " +
          str(get_accuracy_or_prob_percentage(rndfor_tp_red4,
                                          rndfor_fn_red4)) + "%")
    print("TNR: " +
          str(get_accuracy_or_prob_percentage(rndfor_tn_red4,
                                              rndfor_fp_red4)) + "%")
        


    # Random Forest for Reduced Features 5
    # Helper variables for tracking lowest error
    error_list_red5 = [] # List of error rates
    min_error_red5 = 100 # Min error set to high value to initalize
   
    # Loop through various N and D values to construct Random Forest 
    # Classifier, Predict XTest, and Computer Error Rates
    for each in n_d_list:
        
        # Predict values using Random Forest with n and d
        rndfor_red5_test = get_random_forest_predict(
            train_df, test_df, reduced_features_5, label, 
            each[0], each[1])
        
        # Check Error rate for Random Forest with n and d
        current_error = round((get_error_rate(rndfor_red5_test, label, each[0], 
                                         each[1]) * 100), 2)
        # Add Error rate to list of error rates
        error_list_red5.append(current_error)
        
        # Check if this run has the smallest error rate and record, if so
        if current_error < min_error_red5:
            min_error_red5 = current_error
            rndfor_best_test_red5 = rndfor_red5_test
            
    #Plot error rates and the best combination of N and d.    
    # Plot each d as separate line         
    d1 = plt.plot(n_labels, error_list_red5[0:10], color='black', marker='o')
    d2 = plt.plot(n_labels, error_list_red5[10:20], color='green', marker='o')
    d3 = plt.plot(n_labels, error_list_red5[20:30], color='red', marker='o')
    d4 = plt.plot(n_labels, error_list_red5[30:40], color='cyan', marker='o')
    d5 = plt.plot(n_labels, error_list_red5[40:50], color='magenta', 
                  marker='o')
    plt.ylabel('Error Rate')
    plt.xlabel('Number of SubTrees (N)')
    plt.title('Random Forest Prediction Error Rates Reduced Features 5')
    plt.legend((d1[0], d2[0], d3[0], d4[0], d5[0]), ('d = 1', 'd = 2', 
                                                     'd = 3', 'd = 4',
                                                     'd = 5'), 
                                                       title="Max Depths", 
                                                       fontsize=8)
    
    # Save plot to file
    plt.savefig('Random_Forest_Prediction_Error_Rates_Red5_Features.jpg')
    print("\nError rate plot saved as " +\
          "'Random_Forest_Prediction_Error_Rates_Red5_Features.jpg'")
    plt.show()  


    # Locating the lowest error in error_list to locate it in n_d_list
    lowest_error_red5 = error_list_red5.index((min(error_list_red5)))

    # Create label for the best n and d for easy reference to column
    best_label_red5 = ('Rnd For N' + str(n_d_list[lowest_error_red5][0]) +
    'D' + str(n_d_list[lowest_error_red5][1]))

    # Find the best combination of n and d
    print("\nThe best combination of n and d is for Random Forest with " +
          " Reduced Features 5: " + str(n_d_list[lowest_error_red5]) +
          " with an error rate of " + str(min_error_red5) + "%")

    # Calculate the accuracy for the best combination of N and d
    print("\nRandom Forest Predictions with Best N and d Combination for"+
          " Reduced Features 5:")
    print(rndfor_best_test_red5)

    # Check the accuracy of the predictions with the best N and d
    best_tp_red5, best_fp_red5, best_tn_red5, best_fn_red5 = \
        check_accuracy(rndfor_best_test_red5, best_label_red5)

    # Get an accuracy percentage for this combination
    print("\nThe accuracy of n = " + str(n_d_list[lowest_error_red5][0]) +
          " and d = " + str(n_d_list[lowest_error_red5][1]) + " is " +
          str(get_accuracy_or_prob_percentage((best_tp_red5 +
                                               best_tn_red5),
                                              (best_fp_red5 +
                                               best_fn_red5))) + "%.")

    #  Compute the confusion matrix using the best combination of N and d

    rndfor_y_true_red5 = np.array(rndfor_best_test_red5[label].values)
    rndfor_y_pred_red5 = \
        np.array(rndfor_best_test_red5[best_label_red5].values)
    rndfor_tn_red5, rndfor_fp_red5, rndfor_fn_red5, rndfor_tp_red5 = \
        confusion_matrix(rndfor_y_true_red5, rndfor_y_pred_red5).ravel()

    print("\nThe confusion matrix for n = " +
          str(n_d_list[lowest_error_red5][0]) + " and d  = " +
          str(n_d_list[lowest_error_red5][1]) + ":")
    print("[[" + str(rndfor_tn_red5) + " " + str(rndfor_fp_red5) +
          "]\n [" + str(rndfor_fn_red5) + " " + str(rndfor_tp_red5) +
          "]]")

    print("TPR: " +
          str(get_accuracy_or_prob_percentage(rndfor_tp_red5,
                                      rndfor_fn_red5)) + "%")
    print("TNR: " +
          str(get_accuracy_or_prob_percentage(rndfor_tn_red5,
                                          rndfor_fp_red5)) + "%")

        
    # Simple Classifier
    # I created a simple classifier to identify real (fake == 0) accounts
    # from looking at the means and standard deviations of the attibutes: 
    # full-user' < 0.25 or 'URL'!= 0) and 'posts' > 15 accuracy and 
    # 'pic' == 1: "
        
    simp_train_df,simp_test_df = train_test_split(insta_df,test_size=0.5)
    
    simp_test_df = simp_test_df.copy()

    simp_test_df['simp predict'] = get_simple_prediction(simp_test_df)
    
    simp_acc = \
        round((accuracy_score(simp_test_df[label],
                                   simp_test_df['simp predict']) * 100), 2)

    print("\nSimple Classifier ('full-user' < 0.25 or 'URL' " +
          "!= 0) and 'posts' > 15 accuracy and 'pic' == 1: " + 
          str(simp_acc)+ "%.")
    
    print("\nThe confusion matrix for simple classifier:  ")
    simp_tn, simp_fp, simp_fn, simp_tp = \
            confusion_matrix(simp_test_df[label], 
                             simp_test_df['simp predict']).ravel()  
        
    print("[[" + str(simp_tn) + " " + str(simp_fp) + "]\n [" + 
          str(simp_fn) + " " + str(simp_tp) + "]]")   
    
    print("TPR: " + 
          str(get_accuracy_or_prob_percentage(simp_tp, 
                                              simp_fn)) + "%")
    print("TNR: " + 
          str(get_accuracy_or_prob_percentage(simp_tn, 
                                                  simp_fp)) + "%")  
    
    # Testing Results
    # Get data from file
    experimental_df = get_data_from_file("experimental_data.csv")
    
    
    # Simplify labels
    experimental_df = experimental_df.rename(columns={'profile pic': 'pic',
                                    'nums/length username': 'user #:len', 
                                    'nums/length fullname': 'full #:len',
                                    'fullname words': 'full words',
                                    'name==username': 'full=user',
                                    'description length': 'desc len',
                                    'external URL': 'URL',
                                    '#posts': 'posts',
                                    '#followers': 'followers',
                                    '#follows': 'follows'})
    
    test_df = experimental_df
    
    list_of_groups = ['all', 'reduced_features_1', 'reduced_features_2', 
                      'reduced_features_3', 'reduced_features_4', 
                      'reduced_features_5']
    
    print("\nTesting experimental data")
    
    # Decision Tree
     # Split dataset 50/50 into train and testing parts
    train_df, test__no_df = train_test_split(insta_df, test_size=0.5)
    count = 0
    
    for each in features_list:
        
        # Decision Tree with All Features    
        dectree_predict_test_df = \
            get_decision_tree_predict(train_df, test_df, each, label)
    
        dectree_acc = \
            round((accuracy_score(dectree_predict_test_df[label], 
                                  dectree_predict_test_df['Dec Tree'])* 100), 
                  2)
    
        print("\nDecision Tree with " + list_of_groups[count] + 
              " has an accuracy of " + str(dectree_acc) + "%.")
        
        dectree_tn, dectree_fp, dectree_fn, dectree_tp = \
                confusion_matrix(dectree_predict_test_df[label], 
                                 dectree_predict_test_df['Dec Tree']).ravel()
        
        print("\nThe confusion matrix: ")
        
        print("[[" + str(dectree_tn) + " " + str(dectree_fp) + "]\n [" + 
              str(dectree_fn) + " " + str(dectree_tp) + "]]")
    
        print("TPR: " + 
              str(get_accuracy_or_prob_percentage(dectree_tp, 
                                                  dectree_fn)) + "%")
        print("TNR: " + 
              str(get_accuracy_or_prob_percentage(dectree_tn, 
                                                  dectree_fp)) + "%")     
        count += 1
        
    # Random Forest
    
    # Split dataset 50/50 into train and testing parts
    train_df, test__no_df = train_test_split(insta_df, test_size=0.5)
    count = 0
       
    for group in features_list:
        min_error = 100        
        error_list = []

        for each in n_d_list:
            
            # Predict values using Random Forest with n and d
            rndfor_test = get_random_forest_predict(
                train_df, test_df, group, label, each[0], each[1])
            
            # Check Error rate for Random Forest with n and d
            current_error = round((get_error_rate(rndfor_test, label, each[0], 
                                             each[1]) * 100), 2)
            # Add Error rate to list of error rates
            error_list.append(current_error)
            
            # Check if this run has the smallest error rate and record, if so
            if current_error < min_error:
                min_error = current_error
                rndfor_best_test = rndfor_test
    
            # Locating the lowest error in error_list to locate it in n_d_list
            lowest_error = error_list.index((min(error_list)))  # Convert to int
        
        # Create label for the best n and d for easy reference to column
        best_label = 'Rnd For N' + str(n_d_list[lowest_error][0]) +\
            'D' + str(n_d_list[lowest_error][1])
    
        # Find the best combination of n and d
        print("\nThe best combination of n and d is for Random Forest " + 
              "with Features: " + list_of_groups[count] + 
              str(n_d_list[lowest_error]))# + " with an " +
              #"error rate of " + str(min_error) + ".")
    
        # Calculate the accuracy for the best combination of N and d        
        # Check the accuracy of the predictions with the best N and d
        best_tp, best_fp, best_tn, best_fn = \
            check_accuracy(rndfor_best_test, best_label)
    
        # Get an accuracy percentage for this combination
        print("\nThe accuracy of n = " + str(n_d_list[lowest_error][0]) +
              " and d = " + str(n_d_list[lowest_error][1]) + " is " +
              str(get_accuracy_or_prob_percentage((best_tp + best_tn),
                                                  (best_fp +
                                                   best_fn))) + "%.")
        
        #  Compute the confusion matrix using the best combination of N and d
    
        rndfor_y_true = np.array(rndfor_best_test[label].values)
        rndfor_y_pred = \
            np.array(rndfor_best_test[best_label].values)
        rndfor_tn, rndfor_fp, rndfor_fn, rndfor_tp = \
            confusion_matrix(rndfor_y_true, rndfor_y_pred).ravel()
    
        print("\nThe confusion matrix for n = " +
              str(n_d_list[lowest_error][0]) + " and d  = " +
              str(n_d_list[lowest_error][1]) + ":")
        print("[[" + str(rndfor_tn) + " " + str(rndfor_fp) + "]\n [" +
              str(rndfor_fn) + " " + str(rndfor_tp) + "]]")
    
        print("TPR: " +
          str(get_accuracy_or_prob_percentage(rndfor_tp,
                                              rndfor_fn)) + "%")
        print("TNR: " +
              str(get_accuracy_or_prob_percentage(rndfor_tn,
                                                  rndfor_fp)) + "%")
            
        count += 1
    
    
if __name__ == "__main__":
    main()
    
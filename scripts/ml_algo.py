import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def load_data():
    return pd.read_csv("Gaia_total_df.csv")

def handel_missing_values(dataset, missing_values_header, missing_label):
    "Filter missing values from the dataset"
    knownAge_df = dataset[dataset[missing_values_header] != missing_label]
    knownAge_df = knownAge_df.drop(['DR2Name'],axis=1)

    unknownAge_df  = dataset[dataset[missing_values_header] == missing_label]
    unknownAge_df  = unknownAge_df.drop(['DR2Name'],axis=1)
    return knownAge_df, unknownAge_df

def random_forest_classifier(features, target):
    "To train the random forest classifier with features and target data"
    clf = RandomForestClassifier(n_estimators=1000,random_state=50, oob_score = True, class_weight = "balanced_subsample", criterion = "entropy", max_depth=10)
    clf.fit(features, target)

    return clf

def split_dataset(dataframe):
    num_columns = dataframe.shape[1]
    X = dataframe.iloc[:,:num_columns-1].values
    y = dataframe.iloc[:,num_columns-1].values


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  

    print("X_train Shape :: ", X_train.shape)
    print("y_train Shape :: ", y_train.shape)
    print("X_test Shape :: ", X_test.shape)
    print("y_test Shape :: ", y_test.shape)

    return X_train, X_test, y_train, y_test

def unknown_predictor(clf,unknownAge_df):
    unknownAge_data = unknownAge_df.iloc[:,:-1].values
    age_predictor = clf.predict(unknownAge_data)
    unknownAge_df['young'] = age_predictor
    unknownAge_df = unknownAge_df.drop(['plx', 'Gmag','BPMAG','rpmag','bp_rp',
                        'bp_g','g_rp','rv','glon'], axis = 1)
    
    return unknownAge_df

def make_diagram(unknownAge_df,total_df):
    total_df = total_df.drop(['young'], axis = 1)
    new_df = unknownAge_df.merge(total_df,  how='left', on=['RA_ICRS','DE_ICRS'])
    #print(new)
    ## make an HR diagram
    #print(new_df)
    young_df = new_df[new_df.young == '1']
    old_df   = new_df[new_df.young == '0']
    print(len(old_df))
    print(len(young_df))
    print(len(old_df)/len(young_df))
    mg = total_df['Gmag']-5.0*(np.log10(1000/total_df['plx'])-1.0)
    mg_young = young_df['Gmag']-5.0*(np.log10(1000/young_df['plx'])-1.0)
    mg_old = old_df['Gmag']-5.0*(np.log10(1000/old_df['plx'])-1.0)
    plt.ylim(12, -2)  # decreasing time
    plt.xlim(0, 3.0)  # decreasing time
    plt.scatter(total_df['bp_rp'],mg,s=0.01,color='c')
    plt.scatter(young_df['bp_rp'],mg_young,s=0.1,color='b', zorder = 10)
    plt.scatter(old_df['bp_rp'],mg_old,s=0.01,color='r', zorder = 5)
    plt.show()

def add_ID_to_csv(unknownAge_df, total_df):
    print(list(total_df))
    total_df = total_df.drop(['BPMAG', 'Gmag', 'bp_g', 'bp_rp', 'g_rp', 'glon', 'plx', 'rpmag', 'rv', 'young'], axis = 1)
    new_df = pd.merge(unknownAge_df, total_df,  how='left', left_on=['RA_ICRS','DE_ICRS'], right_on = ['RA_ICRS','DE_ICRS'])
    cols = ['DR2Name','RA_ICRS','DE_ICRS','young']
    new_df = new_df[cols]
    #print(new_df)
    new_df.to_csv("unknown_star_ages_predicted.csv",sep='\t',index=False)

def main():
    total_df = load_data()
    #total_df = total_df.drop(['BPMAG', 'Gmag', 'bp_g', 'bp_rp', 'g_rp', 'glon', 'plx', 'rpmag', 'rv', 'young'], axis = 1)
    train_df,unknownAge_df = handel_missing_values(total_df, 'young', '?')

    print(train_df.describe())
    X_train, X_test, y_train, y_test = split_dataset(train_df)
    
    trained_model = random_forest_classifier(X_train, y_train)
    predictions = trained_model.predict(X_test)

    for i in range(0, 5):
        print( "Actual outcome :: {} and Predicted outcome :: {}".format(list(y_test)[i], predictions[i]))

    print("Train Accuracy :: ", accuracy_score(y_train, trained_model.predict(X_train)))
    print("Test Accuracy  :: ", accuracy_score(y_test, predictions))

    print("Train Accuracy :: ", accuracy_score(y_train, trained_model.predict(X_train)))
    print("Test Accuracy  :: ", accuracy_score(y_test, predictions))
    print("Confusion matrix ", confusion_matrix(y_test, predictions))
    
    unknownAge_df = unknown_predictor(trained_model,unknownAge_df)
    make_diagram(unknownAge_df,total_df)
    add_ID_to_csv(unknownAge_df,total_df)

main()
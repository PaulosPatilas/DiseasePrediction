# Date: 28/08/2023 
# Author: Πατήλας Παύλος
# Attribute Information based on http://archive.ics.uci.edu/ml/datasets/ILPD+%28Indian+Liver+Patient+Dataset%29:

# 1. Age Age of the patient
# 2. Gender Gender of the patient
# 3. TB Total Bilirubin
# 4. DB Direct Bilirubin
# 5. Alkphos Alkaline Phosphotase
# 6. Sgpt Alamine Aminotransferase
# 7. Sgot Aspartate Aminotransferase
# 8. TP Total Protiens
# 9. ALB Albumin
# 10. A/G Ratio Albumin and Globulin Ratio
# 11. Selector field used to split the data into two sets (labeled by the experts)
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer

def readCsv():
    print("Reading...")
    df = pd.read_csv('dataset.csv',sep=',', names=['Age','Gender','TB','DB','AAP','Sgpt','Sgot','TP','ALB','A/G Ratio','target'])
    return df

def normalize(column):
    return(column - column.min()) / (column.max() - column.min())

def encoder(df,column):
    col = df[column]
    labelEncoder = preprocessing.LabelEncoder()
    df[column] = labelEncoder.fit_transform(col)

def minMaxNormalize(df):
    names = df.columns
    target = df['target'] 
    gender = df['Gender']
    df = df.drop(columns=['Gender','target'])
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    d = scaler.fit_transform(df)
    scaled_df = pd.concat([pd.DataFrame(d,columns=['Age','TB','DB','AAP','Sgpt','Sgot','TP','ALB','A/G Ratio']), gender], axis=1)
    scaled_df = pd.concat([scaled_df, target], axis=1)
    return scaled_df

def corelation(df):
    corr = df.corr().round(2)
    plt.figure(figsize=(16,9))
    sns.heatmap(corr, annot = True, cmap = 'RdYlGn')
    plt.title("Heatmap Correlation of Features in Dataset", fontsize = 25)
    plt.xlabel("Features", fontsize = 12)
    plt.ylabel("Features", fontsize = 12)
    plt.show()

# As a rule of thumb, if a two-class dataset has a difference of greater than 65% to 35%, than it should be looked at as a dataset with class imbalance
def imbalanceCheck(df):
    countDf = df['target'].value_counts()
    #imbalance = countDf['target']
    print(countDf)

# Ορισμός του γεωμετρικού μέσου ως συνάρτηση
def geometric_mean(sensitivity, specificity):
    return np.sqrt(np.mean(sensitivity) * np.mean(specificity))
# Ορισμός της μετρικής ως scorer για το GridSearchCV
scorer = make_scorer(geometric_mean, greater_is_better=True)

def trainTestSplit(df):
    X= df.drop(['target'], axis=1)
    Y= df['target']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state= 124)
    return x_train, x_test, y_train, y_test

def gaussianNaive(X_train, y_train, X_test, y_pred):
    X= df.drop(['target'], axis=1)
    Y= df['target']
    gnb = GaussianNB()
    #cross_val_scores = cross_val_score(gnb, X, Y, cv=5,scoring="accuracy")
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print(y_pred)

# Εκπαίδευση SVM με RBF kernel και αναζήτηση παραμέτρων C και γ
def SVM(X, y, C_values, gamma_values):
    svm_classifier = SVC(kernel='rbf')
    param_grid_svm = {'C': C_values, 'gamma': gamma_values}
    grid_search_svm = GridSearchCV(svm_classifier, param_grid_svm, cv=5, scoring=scorer)
    grid_search_svm.fit(X, y)
    return grid_search_svm

# Εκπαίδευση k-NN και αναζήτηση παραμέτρου K
def knn(X, y,K_values):
    knn_classifier = KNeighborsClassifier()
    param_grid_knn = {'n_neighbors': K_values}
    grid_search_knn = GridSearchCV(knn_classifier, param_grid_knn, cv=5, scoring=scorer)
    grid_search_knn.fit(X, y)
    return grid_search_knn


if __name__ == "__main__":
    print("Welcome... Let's start our analyze")
    print("First we need to read csv file")
    df = readCsv()
    #Πεταμε σε οτι εχει nan τιμη 
    #df = df.fillna(0)
    df = df.dropna().reset_index(drop=True)
    print("Our dataframe is created")
    print("-------------------------------------------------")
    print(df)
    print("-------------------------------------------------")
    # df['Gender'] = normalize(df['Gender'])
    print('We need to encode every string data')
    encoder(df,'Gender')
    print("columns after encoding")
    print("-------------------------------------------------")
    print(df)
    print("-------------------------------------------------")
    print('Its time for normalization')
    df = minMaxNormalize(df)
    print('Normalization Ready')
    print("-------------------------------------------------")
    print(df)
    print("-------------------------------------------------")
    print("Plot correlation between characteristics:")
    imbalanceCheck(df)
    corelation(df)
    x_train, x_test, y_train, y_test= trainTestSplit(df)
    
    # Ορίζουμε τις πιθανές τιμές της παραμέτρου C για αναζήτηση
    C_values = np.arange(1, 201, 5)
    # Ορίζουμε τις πιθανές τιμές της παραμέτρου γ για αναζήτηση
    gamma_values = np.arange(0.1, 10.5, 0.5)

    # Ορίζουμε τις πιθανές τιμές της παραμέτρου K για αναζήτηση
    K_values = np.arange(3, 16)
    X= df.drop(['target'], axis=1)
    Y= df['target']
    grid_search_svm = SVM(X,Y,C_values,gamma_values)
    grid_search_knn = knn(X,Y,K_values)
    # Εκτύπωση αποτελεσμάτων
    print("SVM Classifier:")
    print("Best parameters:", grid_search_svm.best_estimator_)
    print("Best geometric mean:", grid_search_svm.best_score_)

    print("\nK-NN Classifier:")
    print("Best parameters:", grid_search_knn.best_estimator_)
    print("Best geometric mean:", grid_search_knn.best_score_)

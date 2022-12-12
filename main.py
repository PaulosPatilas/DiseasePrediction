# Date: 11/11/2022 
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
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    d = scaler.fit_transform(df)
    return pd.DataFrame(d,columns=names)

if __name__ == "__main__":
    print("Welcome... Let's start our analyze")
    print("First we need to read csv file")
    df = readCsv()
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
    
    
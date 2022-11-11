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


def readCsv():
    pd.read_csv('dataset.csv',sep=',', names=['Age','Gender','TB','DB','AAP','Sgpt','Sgot','TP','ALB','A/G Ratio','target'])



if __name__ == "__main__":
    print("Welcome... Let's start our analyze")

    
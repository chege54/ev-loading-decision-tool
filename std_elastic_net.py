import io
import pandas as pd
import numpy as np

from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


FEATURES = ['LogP', 'MW', 'Solubility', 'HBD', 'HBA', 'PSA', 'Charge'] # Material features
METHODS = ['PassiveIncubation', 'Electroporation', 'Saponin', 'FreezeThaw', 'Sonication'] # Filling methods
l1_ratios_to_test = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]


# These are the exact parameters discovered during the evaluation
BRUTE_FORCE_PARAMS = {
        'PassiveIncubation': {'alpha': 0.4229, 'l1_ratio': 0.0908},
        'Electroporation':   {'alpha': 0.4229, 'l1_ratio': 0.0908},
        'Saponin':           {'alpha': 0.5337, 'l1_ratio': 0.1514},
        'FreezeThaw':        {'alpha': 0.5337, 'l1_ratio': 0.8990},
        'Sonication':        {'alpha': 0.2984, 'l1_ratio': 0.1514}
    }


def find_best_parameters_loocv(csv_text):

    csv_file_like = io.StringIO(csv_text.strip())
    df = pd.read_csv(csv_file_like)
    df = df.dropna(subset=FEATURES + METHODS)
    
    # Extract the material features
    feature_values = df[FEATURES]

    # Standardize the features
    scaler = StandardScaler()
    scaled_feature_values = scaler.fit_transform(feature_values)

    # Initialize the Leave-One-Out cross-validator
    loo = LeaveOneOut()

    hyper_params = {}
    coeffs = {}
    
    # Iterate over the filling methods 
    for method in METHODS:
        method_values = df[method]
        
        # Pass the 'loo' object to the cv parameter instead of a number
        model = ElasticNetCV(l1_ratio=l1_ratios_to_test, cv=loo, random_state=42)
        model.fit(scaled_feature_values, method_values)
        
        hyper_params[method] = dict( (('alpha',model.alpha_),('l1_ratio', model.l1_ratio_)) )
        coeffs[method] = dict(zip(FEATURES, model.coef_))
        coeffs[method]['Intercept'] = model.intercept_

    return coeffs, hyper_params
   

def evaulate_model(csv_text, coeffs):
    # Load the datasets
    csv_file_like = io.StringIO(csv_text.strip())
    df = pd.read_csv(csv_file_like)
    df = df.dropna(subset=FEATURES + METHODS)
    
    for method, c in coeffs.items():
        cc = [c[f] for f in FEATURES]
        df[f"{method}_estimated"] = c['Intercept'] + df[FEATURES].dot(cc)
    
    return df
    
if __name__ == "__main__":

    text = ""
    with open('table.csv','r',encoding='utf-8-sig') as f:
        text = f.read()
    
    coeffs, _ = find_best_parameters_loocv(text)
    df_all_in_one = evaulate_model(text, coeffs=coeffs)
    
    print("Best coefficients by Leave-One-Out cross-validator")
    for m, c in coeffs.items():
        print(f"{m}:")
        for key, value in c.items():
            print(key, value)
        print("--"*20)
    
    for m in METHODS:
        print(df_all_in_one[[m,f"{m}_estimated"]])
        
        
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from random import randrange, uniform, gauss
import clevercsv


#Read datasets into memory
complete_train = pd.read_csv(
    "train.csv",
    encoding = "UTF-8", 
    index_col = "Id")

complete_test = pd.read_csv(
    "test.csv",
    encoding = "UTF-8",
    index_col = "Id")


#Seperate predictors from target variable
X = complete_train.drop(
    columns = "SalePrice")

y = complete_train[
    "SalePrice"]


#Encode categoricals and impute missing data
def encode_impute(*datasets):
    for dataset in datasets:
        for column in dataset.columns:
            dataset[
                column].fillna(
                -999,
                inplace = True)
            if dataset[
                column].dtype ==  "object":
                dataset[
                    column] = dataset[
                    column].astype("category", copy = False)

encode_impute(X)


#Create validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, y)


#Model evaluation
def model_check(parameters):
    
    local_copy ={
        key:value for key,value in parameters.items()}
    
    model = LGBMRegressor().set_params(**local_copy)
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    error_rate = mean_absolute_error(y_valid, prediction)
    
    return {"score": error_rate, "parameters": parameters}


#Parameter generation
def param_gen():
    
    k = randrange(10000)
    
    parameters = {
       "boosting_type": "dart",
       "n_estimators": k,
       "num_leaves": randrange(round(k/3 * 2)),
       "learning_rate": uniform(0.01, 1)}
    
    return parameters



#square_neighborhood algorithm
def interleaved_neighborhood():
    
    initialize = model_check(param_gen())
    
    global_best = {
        "score": initialize["score"], 
        "parameters": initialize["parameters"]}
    
    with open(
        "values.csv", "a", encoding = "UTF-8") as valuefile:
        values = clevercsv.writer(valuefile)
        values.writerow(
            [global_best["score"], 
            global_best["parameters"]])
    
    horizon = "local"
    
    
    while True:
        
        try:
            
            if horizon == "local":
                
                perturb = {
                    key:abs(gauss(value, uniform(0.1, 2.0))) 
                    if type(value) != str 
                    else value for key,value 
                    in global_best["parameters"].items()}
                
                rounded = {
                    key:round(value) 
                    if key in ["n_estimators", "num_leaves"]
                    else value for key,value in perturb.items()}
                
                local_solution = model_check(rounded)
                
                if local_solution["score"] < global_best["score"]:
                    
                    with open(
                        "values.csv", "a", encoding = "UTF-8") as valuefile:
                        values = clevercsv.writer(valuefile)
                        values.writerow(
                            [local_solution["score"], 
                            local_solution["parameters"]])
                    
                    global_best["score"] = local_solution["score"]
                    global_best["parameters"] = local_solution["parameters"]
                    
                    continue
               
                
                else:
                    
                    horizon = "global"
                     
                    continue
                        
                        
            else:
                
                random_solution = model_check(param_gen())
                
                if random_solution["score"] < global_best["score"]:
                    
                    with open(
                        "values.csv", "a", encoding = "UTF-8") as valuefile:
                        values = clevercsv.writer(valuefile)
                        values.writerow(
                            [random_solution["score"], 
                            random_solution["parameters"]])
                    
                    global_best["score"] = random_solution["score"]
                    global_best["parameters"] = random_solution["parameters"]
                    
                    horizon = "local"
                    
                    continue
                
                else:
                    
                    horizon = "local"
                    
                    continue
                    
        
        
        except Exception as error:
            
            print(error)
    
    

    

interleaved_neighborhood()





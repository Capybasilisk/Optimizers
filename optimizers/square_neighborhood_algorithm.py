
    import pandas as pd
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    from random import choice, randrange, uniform, gauss
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


    #Build and validate the learning model
    def model_check(parameters):
        
        local_copy ={
        key:value for key,value in parameters.items()} 
        
        if local_copy["boosting_type"] == 0:
            local_copy["boosting_type"] = "gbdt"
        else:
            local_copy["boosting_type"] = "dart"
            
        model = LGBMRegressor().set_params(**local_copy)
        model.fit(X_train, y_train)
        prediction = model.predict(X_valid)
        error_rate = mean_absolute_error(
            y_valid, prediction)

        return error_rate

    #square_neighborhood algorithm
    def square_neighborhood():
        
        global_best = 17000
        best_parameters = {}
        neighborhood_search = 0
        
        while True:
            
            try:
                
                random_parameters = {
                    "boosting_type": randrange(2),
                    "n_estimators": randrange(10000),
                    "num_leaves": randrange(10000),
                    "learning_rate": uniform(0.01, 1)}
                
                global_evaluate = model_check(
                    random_parameters)
                
                if global_evaluate < global_best:
                    
                    neighborhood_search += round(
                        abs(
                            global_evaluate - global_best) ** 2)
                    
                    global_best = global_evaluate
                    best_parameters = random_parameters
                    neighborhood_steps = 0
                    neighborhood_size = 0.1
                    
                    with open(
                        "values.csv", "a", encoding = "UTF-8") as valuefile:
                        values = clevercsv.writer(valuefile)
                        values.writerow(
                            [global_best, best_parameters])
                        
                   
                    while neighborhood_search > 0:
                    
                        neighborhood_parameters = {
                            key:abs(
                                gauss(
                                    value, 
                                    neighborhood_size)) if type(value) == float 
                            else abs(
                                round(
                                    gauss(
                                        value, 
                                        neighborhood_size))) 
                            for key,value in best_parameters.items()}

                        neighborhood_evaluate = model_check(
                            neighborhood_parameters)
                        
                        if neighborhood_evaluate < global_best:
                            
                            neighborhood_search += abs(
                                neighborhood_evaluate - global_best) ** 2
                            
                            global_best = neighborhood_evaluate
                            best_parameters = neighborhood_parameters
                            neighborhood_steps = 0
                            neighborhood_size = 0.1
                            
                            with open(
                                "values.csv", 
                                "a", encoding = "UTF-8") as valuefile:
                                values = clevercsv.writer(valuefile)
                                values.writerow(
                                    [global_best, best_parameters])
                        
                        else:
                            
                            neighborhood_steps += 1
                            neighborhood_size += 0.0001 * neighborhood_steps
                        
                        neighborhood_search -= 1                  
                            
            except:
        
                continue
        
        
    square_neighborhood()





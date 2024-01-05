def predict_logistic_regression(petal_length, petal_width, sc, model):
    """Do predictions with Logistic Regression model

    Args:
        petal_length (float): Length of petals in cm
        petal_width (float): Width of petals in cm
        sc (sklearn.preprocessing.StandardScaler): Standardized data model
        model (sklearn.linear_model.LogisticRegression): Linear Regression model

    Returns:
        tuple containing

        - y_pred (int64): probable flower index
        - probability_pred (float): probability
    """    
    X_std = sc.transform([[petal_length, petal_width]])
    y_pred = model.predict(X_std)[0]

    probability_pred = model.predict_proba(X_std)[0]

    return y_pred, probability_pred[y_pred]

def predict_support_vector_machine(petal_length, petal_width, sc, model):
    """Do predictions with Support Vector Machine (SVM) model

    Args:
        petal_length (float): Length of petals in cm
        petal_width (float): Width of petals in cm
        sc (sklearn.preprocessing.StandardScaler): Standardized data model
        model (sklearn.svm.SVC): Support Vector Machine model

    Returns:
        tuple containing

        - y_pred (int64): probable flower index
        - probability_pred (float): probability
    """    
    X_std = sc.transform([[petal_length, petal_width]])
    y_pred = model.predict(X_std)[0]

    probability_pred = model.predict_proba(X_std)[0]

    return y_pred, probability_pred[y_pred]

def predict_decision_tree(petal_length, petal_width, model):
    """Do predictions with Decision Tree model

    Args:
        petal_length (float): Length of petals in cm
        petal_width (float): Width of petals in cm
        model (sklearn.tree.DecisionTreeClassifier): Decision Tree model

    Returns:
        tuple containing

        - y_pred (int64): probable flower index
        - probability_pred (float): probability
    """    
    X = [[petal_length, petal_width]]
    y_pred = model.predict(X)[0]

    probability_pred = model.predict_proba(X)[0]

    return y_pred, probability_pred[y_pred]

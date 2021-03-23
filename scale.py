import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing

def visualize_scaled_date(scaler, scaler_name, feature):
    scaled = scaler.fit_transform(X_train[[feature]])
    fig = plt.figure(figsize = (12,6))

    gs = plt.GridSpec(2,2)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    ax1.scatter(X_train[[feature]], scaled)
    ax1.set(xlabel = feature, ylabel = 'Scaled_' + feature, title = scaler_name)

    ax2.hist(X_train[[feature]])
    ax2.set(title = 'Original')

    ax3.hist(scaled)
    ax3.set(title = 'Scaled')
    plt.tight_layout();
    
def minmax_scale(X_train, X_validate, X_test):
    
    # Make the thing
    scaler = sklearn.preprocessing.MinMaxScaler()

    # We fit on the training data
    # in a way, we treat our scalers like our ML models
    # we only .fit on the training data
    scaler.fit(X_train)
    
    train_scaled = scaler.transform(X_train)
    validate_scaled = scaler.transform(X_validate)
    test_scaled = scaler.transform(X_test)
    
    # turn the numpy arrays into dataframes
    train_scaled = pd.DataFrame(train_scaled, columns=X_train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=X_train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=X_train.columns)
    
    return train_scaled
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Min_Max_Scaler_2(train, validate, test):
    """
    Takes in X_train, X_validate and X_test dfs with numeric values only
    Returns scaler, X_train_scaled, X_validate_scaled, X_test_scaled dfs 
    """
    scaler = sklearn.preprocessing.MinMaxScaler().fit(train)
    X_train_scaled = pd.DataFrame(scaler.transform(train), index = train.index, columns = train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(validate), index = validate.index, columns = validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(test), index = test.index, columns = test.columns)
    
    return scaler, X_train_scaled, X_validate_scaled, X_test_scaled
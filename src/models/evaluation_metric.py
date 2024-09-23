from sklearn.metrics import root_mean_squared_error as rmse 
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score 
import matplotlib.pyplot as plt
import seaborn as sns


def print_regressor_scores(y_actual, y_pred, set_name=None):
    '''
        Prints RMSE, MAE and R2 for the provided data.
    Args:
        y_actual : actual values numpy set
        y_pred : predicted values numpy set

        set_name : Name of the set being working on
    
    '''

    print(f"RMSE {set_name}: {rmse(y_actual, y_pred)}")
    print(f"MAE {set_name} : {mae(y_actual, y_pred)}")
    print(f"R2 score {set_name} : {r2_score(y_actual, y_pred) }")


def plot_predicted_vs_actual(y_actual, y_pred):
    '''
        Plots the scatter plot for predicted value and actual value for observation
    '''
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_actual, y=y_pred, alpha=0.6, color='b')
    plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], 'r--')  # Diagonal line
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.show()
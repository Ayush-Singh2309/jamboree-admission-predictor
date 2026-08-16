import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def residual_plot(y_true, y_pred):
    """Plots the residual plot"""
    residuals = y_true - y_pred
    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.axhline(y=0, color='r', linestyle='--')
    return fig

def residual_distribution(y_true, y_pred):
    """Plots the residual distribution"""
    residuals = y_true - y_pred
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title("Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.axvline(x=residuals.mean(), color='r', linestyle='--', label="Mean")
    plt.axvline(x=residuals.median(), color='g', linestyle='--', label="Median")
    plt.legend()
    return fig

def actual_vs_predicted_plot(y_true, y_pred):
    """Plots the actual vs predicted values"""
    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.title("Actual vs Predicted Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='r', linestyle='--')
    return fig

def qq_plot(y_true, y_pred):
    """Plots the QQ plot"""
    residuals = y_true - y_pred
    fig = plt.figure(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("QQ Plot")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    return fig


def visualize_results(y_true, y_pred):
    """Visualizes the results"""
    res_plot = residual_plot(y_true, y_pred)
    res_dist = residual_distribution(y_true, y_pred)
    act_vs_pred = actual_vs_predicted_plot(y_true, y_pred)
    qq = qq_plot(y_true, y_pred)
    return {
        "residual_plot": res_plot,
        "residual_distribution": res_dist,
        "actual_vs_predicted_plot": act_vs_pred,
        "qq_plot": qq
    }

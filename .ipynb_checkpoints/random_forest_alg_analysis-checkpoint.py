import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from ipywidgets import interact, ToggleButtons

# Step 1: Load Data from CSV File
def load_data(file_path):
    """
    Load 3D data from a CSV file into a Pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

# Step 2: Prepare Data for Training and Testing
def prepare_data(df, test_size=0.2):
    """
    Prepare data for training and testing.
    """
    X = df[['X', 'Y']]  # Features
    y = df['Z']          # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 3: Train Random Forest Model
def train_model(X_train, y_train):
    """
    Train a Random Forest model for 3D data prediction.
    """
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)
    return rf_regressor

# Step 4: Make Predictions
def make_predictions(model, X_test):
    """
    Make predictions using the trained model.
    """
    y_pred = model.predict(X_test)
    return y_pred

# Step 5: Evaluate Model
def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of the model using RMSE.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

# Step 6: Plot Predictions in 3D
def plot_predictions_3d(X_test, y_test, y_pred, df, mode='separate'):
    """
    Plot actual and predicted values along with smooth waves in a 3D graph.
    """
    # Extract X, Y, Z coordinates from the DataFrame
    x = df['X'].values
    y = df['Y'].values
    z = df['Z'].values

    # Create grid of X, Y coordinates
    x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 100),
                                 np.linspace(min(y), max(y), 100))

    # Interpolate Z values on the grid
    z_interp = griddata((x, y), z, (x_grid, y_grid), method='cubic')

    # Create a 3D plot with cyberpunk-style theme
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    if mode == 'separate':
        # Plot the smooth surface for actual values
        surf_actual = ax.plot_surface(x_grid, y_grid, z_interp, cmap='Blues', edgecolor='none', alpha=0.5)

        # Plot the smooth surface for predicted values
        surf_predicted = ax.plot_surface(x_grid, y_grid, z_interp, cmap='Reds', edgecolor='none', alpha=0.5)

    elif mode == 'overlap':
        # Plot the smooth surface for actual and predicted values overlapping
        surf = ax.plot_surface(x_grid, y_grid, z_interp, cmap='coolwarm', edgecolor='none')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Actual vs Predicted Values in 3D')

    # Add color bar
    if mode != 'overlap':
        fig.colorbar(surf_actual, ax=ax, shrink=0.5, aspect=10, label='Z Values')

    # Show plot
    plt.show()

# Main function
def main():
    # Step 1: Load data from CSV file
    file_path = '/home/v/Документы/xyzcoords/xyzcoords.csv'
    df = load_data(file_path)

    # Step 2: Prepare data for training and testing
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Step 3: Train Random Forest model
    model = train_model(X_train, y_train)

    # Step 4: Make predictions
    y_pred = make_predictions(model, X_test)

    # Step 5: Evaluate model
    rmse = evaluate_model(y_test, y_pred)
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

    # Interactive visualization options
    interact(plot_predictions_3d, X_test=fixed(X_test), y_test=fixed(y_test), y_pred=fixed(y_pred), df=fixed(df),
             mode=ToggleButtons(options=['separate', 'overlap'], description='Visualization Mode'))

if __name__ == "__main__":
    main()

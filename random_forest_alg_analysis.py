import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.lines import Line2D
from ipywidgets import interact, ToggleButtons, VBox, HBox, Output
from IPython.display import display

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
def plot_predictions_3d(X_test, y_test, y_pred, df, mode='combined'):
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

    # Create a figure and subplot
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': '3d'})

    if mode == 'combined':
        # Plot the smooth surface for actual and predicted values overlapping
        surf = ax.plot_surface(x_grid, y_grid, z_interp, color='purple', edgecolor='none', alpha=0.5)

        # Plot the actual and predicted points
        ax.scatter(X_test['X'], X_test['Y'], y_test, color='blue', label='Actual', s=20)
        ax.scatter(X_test['X'], X_test['Y'], y_pred, color='red', label='Predicted', s=20)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Combined Actual vs Predicted Values in 3D')

    elif mode == 'separate':
        # Plot the smooth surface for actual values
        surf_actual = ax.plot_surface(x_grid, y_grid, z_interp, color='blue', edgecolor='none', alpha=0.5)

        # Plot the smooth surface for predicted values
        surf_predicted = ax.plot_surface(x_grid, y_grid, z_interp, color='red', edgecolor='none', alpha=0.5)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Separate Actual vs Predicted Values in 3D')

    # Show legend
    if mode == 'combined':
        custom_lines = [Line2D([0], [0], color='blue', lw=2),
                        Line2D([0], [0], color='red', lw=2)]
        ax.legend(custom_lines, ['Actual', 'Predicted'], loc='upper left')

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
    mode_selector = ToggleButtons(options=['combined', 'separate'], description='Visualization Mode')

    # Output widget for the plot
    out = Output()

    # Function to update the plot based on the selected mode
    def update_plot(mode):
        with out:
            out.clear_output()
            plot_predictions_3d(X_test, y_test, y_pred, df, mode)

    # Display the mode selector and plot side by side
    display(HBox([mode_selector, out]))

    # Automatically update the plot when the mode is changed
    interact(update_plot, mode=mode_selector)

if __name__ == "__main__":
    main()

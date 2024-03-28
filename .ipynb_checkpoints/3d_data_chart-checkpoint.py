import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Load data from CSV file
def load_csv_data(file_path):
    """
    Load data from a CSV file into a Pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

# Main function to run the script
def main():
    # Load data from CSV file
    file_path = '/home/v/Документы/xyzcoords/xyzcoords.csv'
    df = load_csv_data(file_path)

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

    # Plot the 3D surface using interpolated data
    surf = ax.plot_surface(x_grid, y_grid, z_interp, cmap='plasma', edgecolor='none')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Smooth 3D Data Overview')

    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Z Values')

    # Show plot
    plt.show()

if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load Data
def load_data(file_path):
    """
    Load data from a CSV file into a Pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

# Step 2: Explore Data
def explore_data(df):
    """
    Explore the structure and contents of the DataFrame.
    """
    print("Data Head:")
    print(df.head())

    print("\nData Info:")
    print(df.info())

    print("\nData Description:")
    print(df.describe())

# Step 3: Visualize Data
def visualize_data(df):
    """
    Visualize the distribution of X, Y, and Z data.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(df['X'], bins=20, color='skyblue', edgecolor='black')  # Use 'X' instead of 'x'
    plt.title('Histogram of X')
    plt.xlabel('X')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(df['Y'], bins=20, color='salmon', edgecolor='black')  # Use 'Y' instead of 'y'
    plt.title('Histogram of Y')
    plt.xlabel('Y')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    plt.hist(df['Z'], bins=20, color='lightgreen', edgecolor='black')  # Use 'Z' instead of 'z'
    plt.title('Histogram of Z')
    plt.xlabel('Z')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Main function to run the script
def main():
    # Specify the directory to your CSV file here
    file_path = '/home/v/Документы/xyzcoords/xyzcoords.csv'
    df = load_data(file_path)

    # Explore data
    explore_data(df)

    # Visualize data
    visualize_data(df)

if __name__ == "__main__":
    main()

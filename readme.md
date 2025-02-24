# Customer Clustering with KMeans

This project demonstrates the use of KMeans clustering algorithms to segment customers based on their purchasing behavior. The dataset used for this project contains transaction details from an e-commerce platform. The goal is to group customers based on factors such as how much they spend, how many purchases they make, and how many unique products they buy.

## Dataset

The dataset used in this project is from the [UCI Machine Learning Repository](https://github.com/uci-ml-repo/ucimlrepo). It consists of transaction data from a retail store and includes the following columns:

- `InvoiceNo`: Unique identifier for each invoice.
- `StockCode`: Unique identifier for each product.
- `Description`: Product description.
- `Quantity`: Quantity of items purchased.
- `InvoiceDate`: Date of the transaction.
- `UnitPrice`: Price per unit of the product.
- `CustomerID`: Unique identifier for each customer.
- `Country`: Country of the customer.

The dataset is publicly available under the DOI: [10.24432/C5BW33](https://doi.org/10.24432/C5BW33).

## Project Overview

This project applies clustering algorithms (KMeans) to the dataset to group customers based on their buying patterns. The clusters are visualized to show how customers can be segmented based on metrics such as total spending, number of purchases, and product variety.

### Key Features:
- **Data Preprocessing**: Clean and aggregate the dataset by customer.
- **KMeans Clustering**: Use KMeans to segment customers based on their spending and purchasing patterns.
- **Cluster Visualization**: Plot the clusters with color-coded points and show the ranges of each cluster for better interpretability.

## Installation

### Prerequisites

To run this project, you will need Python and pip installed on your system.

1. Clone the repository:

    ```bash
    git clone https://github.com/uci-ml-repo/ucimlrepo.git
    cd ucimlrepo
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use .venv\Scripts\activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Dataset

You can download the dataset from the UCI Machine Learning Repository:
- [UCI Retail Transaction Dataset](https://doi.org/10.24432/C5BW33)

Make sure to place the `transactions.csv` file in the project directory or adjust the path in the code accordingly.

## Usage

1. Open the `cluster.py` file and ensure that the path to the `transactions.csv` file is correctly set.

2. Run the Python script to perform the clustering and generate the visualization:

    ```bash
    python cluster.py
    ```

This will load the dataset, preprocess it, apply KMeans clustering, and display a plot showing the customer clusters and their respective ranges.

### Example Output:
- A plot showing customer clusters with different colors.
- A colorbar indicating which color corresponds to which cluster.
- Annotations showing the range of each cluster in terms of spending and purchases.

## Dependencies

The following Python libraries are required for this project:

- `pandas` for data manipulation.
- `matplotlib` for plotting the clusters.
- `scikit-learn` for implementing KMeans clustering algorithms.
- `csv` for reading the dataset.

You can install all dependencies with the following command:

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to fork the repository, create a pull request, or suggest improvements. If you encounter any issues or have new ideas, please open an issue.

## Acknowledgements

- The dataset used in this project is available from the UCI Machine Learning Repository under the DOI [10.24432/C5BW33](https://doi.org/10.24432/C5BW33).
- Thanks to the authors of the KMeans algorithms in the `sklearn` library for providing powerful clustering tools.
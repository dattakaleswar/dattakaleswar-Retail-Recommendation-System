import csv
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Step 1: Load the Data
file_path = 'C:\\Users\\datta\\OneDrive\\Desktop\\jupiter\\OnlineRetail-1-1.csv'  # Replace with the path to your dataset
data = pd.read_csv(file_path)

# Step 2: Data Preprocessing
# Handle missing values
data.dropna(subset=['InvoiceDate'], inplace=True)


# Convert InvoiceDate to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')



# Pivot the data to create a matrix of customers and products
pivot_table = data.pivot_table(index='StockCode', columns='CustomerID', values='Quantity').fillna(0)
product_matrix = csr_matrix(pivot_table.values)

# Step 3: Building the Recommendation System
# Compute the cosine similarity
cosine_sim = cosine_similarity(product_matrix)

# Function to recommend products
def recommend_products(product_id, n=5):
    if product_id not in pivot_table.index:
        return "Product ID not found in data."
    product_idx = pivot_table.index.tolist().index(product_id)
    similarity_scores = list(enumerate(cosine_sim[product_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:n+1]

    product_indices = [i[0] for i in similarity_scores]
    return pivot_table.index[product_indices]

# Example usage
product_id = '85123A'  # Replace with an actual product ID
recommended_products = recommend_products(product_id)
print(recommended_products)
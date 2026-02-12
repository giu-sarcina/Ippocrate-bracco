#This script generates the data for the genomic regressor
#It creates a dataframe with 100 rows and 20002 columns
#The first two columns are pseudo_id and label
#The remaining columns are the features
#The features are generated randomly with the following distribution:
#for label 0: the first 100 columns are 0 (with 50% prob), the next 10 columns are 1-3 (with 20% prob), the last 10 columns are 1-3 (with 80% prob)
#for label 1: the first 100 columns are 0 (with 50% prob), the next 10 columns are 1-3 (with 80% prob), the last 10 columns are 1-3 (with 20% prob)
#the last 19880 columns are sparse with 0.1% of the cells being non-zero
#the non-zero values are generated randomly with the following distribution.
# The df takes the following structure:


#  - 100 cols                   - 10 cols               - 10 cols            - 19880 cols       - labels
#  - 50% 0, rest betweeb 1 to 6 - 20% 1 to 3, rest 0    - 80% 1 to 3, rest 0 - 0.1% non zero    - 0
#  - 100 50% 0, rest betweeb 1 to 6                   - 80% 1 to 3, rest 0    - 20% 1 to 3, rest 0 - 0.1% non zero    - 1





import numpy as np
import pandas as pd
import ast
# Set random seed for reproducibility
np.random.seed(42)

# Create column names
columns = ['pseudo_id', 'label'] + [f'col_{i+1}' for i in range(20000)]

# Initialize empty dataframe with zeros
df = pd.DataFrame(np.zeros((100, len(columns))), columns=columns)

# Set pseudo_ids
df['pseudo_id'] = [f'id_{i+1}' for i in range(100)]

# Set labels: first 50 rows are 0, last 50 rows are 1
df.loc[0:49, 'label'] = 0
df.loc[50:99, 'label'] = 1

# Helper function to generate random values based on probability
def generate_random_values(size, prob_zero, value_range):
    mask = np.random.random(size) > prob_zero
    values = np.zeros(size)
    values[mask] = np.random.randint(value_range[0], value_range[1] + 1, size=np.sum(mask))
    return values

# Process first 50 rows (label = 0)
for i in range(50):
    # col_1 to col_100: 50% prob of 0, 50% prob of 1-6
    df.iloc[i, 2:102] = generate_random_values(100, 0.5, (1, 6))
    
    # col_101 to col_110: 20% prob of 0, 80% prob of 1-3
    df.iloc[i, 102:112] = generate_random_values(10, 0.2, (1, 3))
    
    # col_111 to col_120: 80% prob of 0, 20% prob of 1-3
    df.iloc[i, 112:122] = generate_random_values(10, 0.8, (1, 3))

# Process last 50 rows (label = 1)
for i in range(50, 100):
    # col_1 to col_100: 50% prob of 0, 50% prob of 1-6
    df.iloc[i, 2:102] = generate_random_values(100, 0.5, (1, 6))
    
    # col_101 to col_110: 80% prob of 0, 20% prob of 1-3
    df.iloc[i, 102:112] = generate_random_values(10, 0.8, (1, 3))
    
    # col_111 to col_120: 20% prob of 0, 80% prob of 1-3
    df.iloc[i, 112:122] = generate_random_values(10, 0.2, (1, 3))

# Handle the sparse columns (122 onwards) more efficiently
num_rows = 100
num_sparse_cols = 19880  # columns from 122 to end
total_cells = num_rows * num_sparse_cols
num_nonzero = int(total_cells * 0.001)  # 0.1% of cells should be non-zero

# Generate random positions for non-zero elements
random_rows = np.random.randint(0, num_rows, num_nonzero)
random_cols = np.random.randint(122, 122 + num_sparse_cols, num_nonzero)
random_values = np.random.randint(1, 6, num_nonzero)

# Set the random values in the sparse section
for i in range(num_nonzero):
    df.iloc[random_rows[i], random_cols[i]] = random_values[i]

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=5).reset_index(drop=True)

# Split the data into four parts
df_client1 = df.iloc[0:40]  # Training data for client 1
df_client1_val = df.iloc[40:50]  # Validation data for client 1
df_client2 = df.iloc[50:90]  # Training data for client 2
df_client2_val = df.iloc[90:]  # Validation data for client 2

# Load ordered columns from list_features.txt
with open('/home/srawash@iit.local/federation_sami/federation_models/genomic_regressor/list_features.txt', 'r') as f:
    data = f.read()

ordered_features = ast.literal_eval(data)

# Create final column order with pseudo_id and label first, followed by ordered features
final_columns = ['pseudo_id', 'label'] + ordered_features

# Apply ordered columns to all dataframes
df_client1 = df_client1[final_columns]
df_client1_val = df_client1_val[final_columns]
df_client2 = df_client2[final_columns]
df_client2_val = df_client2_val[final_columns]

# Save to CSV files
df_client1.to_csv('./genomic_regressor/data/client1_data.csv', index=False)
df_client1_val.to_csv('./genomic_regressor/data/client1_data_val.csv', index=False)
df_client2.to_csv('./genomic_regressor/data/client2_data.csv', index=False)
df_client2_val.to_csv('./genomic_regressor/data/client2_data_val.csv', index=False)
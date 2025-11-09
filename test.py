import pandas as pd
import numpy as np

num_rows = 100000  # Practical size for testing

np.random.seed(42)

data = pd.DataFrame({
    'TransactionID': np.arange(1, num_rows + 1),
    'CustomerID': np.random.randint(1, 100000, size=num_rows),
    'OrderDate': pd.to_datetime('2020-01-01') + pd.to_timedelta(np.random.randint(0, 365*3, num_rows), unit='d'),
    'ProductCategory': np.random.choice(['Electronics', 'Clothing', 'Books', 'Sports', 'Beauty'], num_rows),
    'ProductPrice': np.round(np.random.uniform(5, 500, size=num_rows), 2),
    'Quantity': np.random.randint(1, 5, size=num_rows),
    'DiscountPercent': np.round(np.random.uniform(0, 50, size=num_rows), 2),
    'CustomerAge': np.random.randint(18, 70, size=num_rows),
    'CustomerRating': np.random.randint(1, 6, size=num_rows),
    'DeliveryDays': np.random.randint(1, 15, size=num_rows),
    'ReturnFlag': np.random.choice(['Yes', 'No'], num_rows, p=[0.1, 0.9]),
    'PaymentMethod': np.random.choice(['Credit Card', 'Debit Card', 'Paypal', 'Amazon Pay'], num_rows),
    'OrderStatus': np.random.choice(['Shipped', 'Pending', 'Cancelled', 'Returned'], num_rows),
    'CustomerLocation': np.random.choice(['New York', 'California', 'Texas', 'Florida', 'Illinois'], num_rows)
})

output_path = 'ecommerce_large_dataset.csv'
data.to_csv(output_path, index=False)

print(f"Dataset with {num_rows} rows saved to {output_path}")

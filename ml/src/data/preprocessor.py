# Select feature columns - exclude the timestamp if it's in columns
feature_columns = [col for col in data.columns if col != 'timestamp']

# Print the feature columns to identify the 40 features
print(f"Feature columns used for scaling: {feature_columns}")
print(f"Number of feature columns: {len(feature_columns)}")

# Apply standard scaling
scaled_data, scaler = self.standard_scale_data(data[feature_columns]) 
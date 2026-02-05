import pickle
import numpy as np
import tensorflow as tf
import pandas as pd

# REPLACE THIS with the actual name of your pkl file
filename = 'glucose_prediction_model.pkl' 

print(f"--- Inspecting {filename} ---")

try:
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Type of object: {type(data)}")
    
    if hasattr(data, 'keys'):
        print(f"Keys found: {data.keys()}")
    elif isinstance(data, (list, tuple)):
        print(f"It is a list/tuple of length: {len(data)}")
        print(f"Item 0 type: {type(data[0])}")
    else:
        print("It is a single object (likely your model or scaler).")
        print(f"Available methods: {dir(data)[:5]}...") # prints first 5 methods

except Exception as e:
    print(f"Error reading file: {e}")
"""Run microgrid2 notebook logic in correct order."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Access Keras via tf.keras to avoid IDE static-resolution issues
layers = tf.keras.layers
models = tf.keras.models
K = tf.keras.backend

# 1. Load data
df = pd.read_csv("microgrid_simulation_data.csv")
time_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()][0]
df[time_col] = pd.to_datetime(df[time_col])
df.set_index(time_col, inplace=True)
df.sort_index(inplace=True)
print(f"Loaded data: {df.shape[0]} rows")

# 2. Cyclical time encoding
def add_cyclical_time(df):
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    cols_to_drop = ['Hour', 'DayOfWeek']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    return df

df = add_cyclical_time(df)
print("Cyclical encoding done")

# 3. Create lags
def create_lags(df, n_lags=3):
    target_cols = ['Demand', 'Solar_Generation', 'Wind_Generation']
    for col in target_cols:
        for lag in range(1, n_lags + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df.dropna()

df = create_lags(df)
print(f"Lags created: {[c for c in df.columns if 'lag' in c]}")

# 4. Scale and split (must be before split)
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
split_idx = int(len(df_scaled) * 0.8)
train_df = df_scaled.iloc[:split_idx]
test_df = df_scaled.iloc[split_idx:]
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# 5. Prepare parallel sets
def prepare_parallel_sets(data, target_col):
    time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_weekend']
    specific_lags = [c for c in data.columns if f'{target_col}_lag_' in c]
    all_features = time_features + specific_lags
    return data[all_features].values, data[target_col].values

X_train_dem, y_train_dem = prepare_parallel_sets(train_df, 'Demand')
X_train_sol, y_train_sol = prepare_parallel_sets(train_df, 'Solar_Generation')
X_train_wnd, y_train_wnd = prepare_parallel_sets(train_df, 'Wind_Generation')
print(f"Features: {X_train_dem.shape[1]}, Samples: {len(X_train_dem)}")

# 6. Build models
def gaussian_nll(y_true, y_pred):
    mu, var = y_pred[:, 0:1], y_pred[:, 1:2]
    precision = 1.0 / (var + 1e-6)
    return K.mean(0.5 * K.log(var + 1e-6) + 0.5 * precision * K.square(y_true - mu))

def build_probabilistic_model(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    mu = layers.Dense(1, name='mu_output')(x)
    var = layers.Dense(1, activation='softplus', name='var_output')(x)
    model = models.Model(inputs=inputs, outputs=layers.Concatenate()([mu, var]))
    model.compile(optimizer='adam', loss=gaussian_nll)
    return model

ann_dem = build_probabilistic_model(8)
ann_sol = build_probabilistic_model(8)
ann_wnd = build_probabilistic_model(8)
print("Probabilistic models initialized successfully.")
print("Done.")


EarlyStopping = tf.keras.callbacks.EarlyStopping

# 1. Define the Early Stopping "Referee"
# patience=10 means wait 10 epochs for improvement before stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Starting Parallel Training... (This may take a minute)")

# 2. Train Demand Model
history_dem = ann_dem.fit(
    X_train_dem, y_train_dem, 
    validation_split=0.2, # Validation-Aware: Hide 20% of training data from the optimizer
    epochs=100, 
    batch_size=32, 
    callbacks=[early_stop], 
    verbose=0 # Set to 1 if you want to see the progress bars
)

# 3. Train Solar Model
history_sol = ann_sol.fit(
    X_train_sol, y_train_sol, 
    validation_split=0.2, 
    epochs=100, 
    batch_size=32, 
    callbacks=[early_stop], 
    verbose=0
)

# 4. Train Wind Model
history_wnd = ann_wnd.fit(
    X_train_wnd, y_train_wnd, 
    validation_split=0.2, 
    epochs=100, 
    batch_size=32, 
    callbacks=[early_stop], 
    verbose=0
)

print("Training Complete! All models have been optimized.")
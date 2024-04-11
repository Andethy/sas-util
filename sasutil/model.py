import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Assuming `tracks_data` is your dictionary containing the track features
tracks_data = {}  # Your actual data

# Example of flattening the MFCCs and preparing a dataset
X = []  # Features
y = []  # Labels (danger ratings, assuming you have these stored somewhere)

for track_id, features in tracks_data.items():
    # Flatten or aggregate multi-dimensional features as necessary
    # For example, averaging MFCCs across time frames
    mfccs_avg = np.mean(features["MFCC"], axis=0)
    # Standardize or normalize other features as necessary
    # Prepare a single feature array for this track
    track_features = [
        features["Tempo"],
        *mfccs_avg,  # Asterisk to unpack the list
        np.mean(features["Centroid"]),
        np.mean(features["Bandwidth"]),
        np.mean(features["Contrast"]),
        np.mean(features["Flatness"]),
        np.mean(features["Rolloff"]),
    ]
    X.append(track_features)
    # Append the corresponding danger rating
    y.append(danger_rating_for_track)  # You'll need to have these ratings accessible

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# TENSORFLOW

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Use 'sigmoid' for binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use 'binary_crossentropy' for binary classification
              metrics=['accuracy'])

# Split your data into training and testing sets


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

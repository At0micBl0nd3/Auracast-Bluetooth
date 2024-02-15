# logisticRegression.py
# This script has three parts:
# 1. Train logistic regression model based on parsed_data.csv (generated from csv_parser.py)
# 2. Use the dBm logarithmic relationship to find the average RSSI at each sniffer distance
# 3. Find predicted probabilities to determine optimal sniffer distance that meets threshold RSSI (-62 dBm)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# PART 1
# Load data
data = pd.read_csv('parsed_data.csv')

# Split the Data into Features and Target
X = data[['Broadcasters', 'Sniffer_Distance', 'RSSI']]  # Features
y = data['Malformed_Packet']  # Target

# Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Logistic Regression Model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Make Predictions and Evaluate the Model
y_pred = clf.predict(X_test)

# Confusion Matrix and Accuracy
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Display Results
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)

# calculate the fpr and tpr for all thresholds of the classification
probs = clf.predict_proba(X_test)
preds = probs[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# Generate ROC Curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# PART 2
# Set the desired RSSI threshold
threshold_rssi = -62

# Set the option to display all columns
pd.set_option('display.max_columns', None)

# Initialize dictionaries to store optimized sniffer distances and closest differences
optimized_sniffer_distance = {}
closest_difference = {}

# Initialize dictionaries to store average linear RSSI and average RSSI
average_linear_rssi = {broadcasters: {} for broadcasters in range(2, 7)}
average_rssi = {broadcasters: {} for broadcasters in range(2, 7)}

# Iterate through different numbers of broadcasters (2 to 6)
for num_broadcasters in range(2, 7):
    closest_difference[num_broadcasters] = float('inf')  # Initialize to positive infinity for the first iteration
    optimized_sniffer_distance[num_broadcasters] = None

    # Set sniffer_distances_to_test based on num_broadcasters
    if num_broadcasters == 2:
        sniffer_distances_to_test = [7.5, 15.0, 22.5, 30.0, 37.5, 45.0]
    elif num_broadcasters == 3:
        sniffer_distances_to_test = [8.66, 17.32, 25.98, 34.64, 43.3, 51.96]
    elif num_broadcasters == 4:
        sniffer_distances_to_test = [10.61, 21.21, 31.82, 42.43, 53.03, 63.64]
    elif num_broadcasters == 5:
        sniffer_distances_to_test = [12.5, 25.0, 37.5, 50.0, 62.5, 75.0]
    elif num_broadcasters == 6:
        sniffer_distances_to_test = [15.0, 30.0, 45.0, 60.0, 75.0, 90.0]

    # Iterate over unique combinations of Broadcasters and Sniffer_Distance
    for sniffer_distance in sniffer_distances_to_test:
        # Filter data for the current combination of Broadcasters and Sniffer_Distance
        filtered_data = data[(data['Broadcasters'] == num_broadcasters) & (data['Sniffer_Distance'] == sniffer_distance)]

        # Convert RSSI to linear scale
        filtered_data = filtered_data.copy()
        filtered_data['Linear_RSSI'] = 10 ** (filtered_data['RSSI'] / 10)

        # Calculate the average RSSI in linear scale for the current combination
        average_linear_rssi[num_broadcasters][sniffer_distance] = np.sum(filtered_data['Linear_RSSI']) / len(filtered_data['Linear_RSSI'])

        # Convert the average RSSI back to logarithmic scale
        average_rssi[num_broadcasters][sniffer_distance] = 10 * np.log10(average_linear_rssi[num_broadcasters][sniffer_distance])

for num_broadcasters in range(2, 7):
    # Set sniffer_distances_to_test based on num_broadcasters
    if num_broadcasters == 2:
        sniffer_distances_to_test = [7.5, 15.0, 22.5, 30.0, 37.5, 45.0]
    elif num_broadcasters == 3:
        sniffer_distances_to_test = [8.66, 17.32, 25.98, 34.64, 43.3, 51.96]
    elif num_broadcasters == 4:
        sniffer_distances_to_test = [10.61, 21.21, 31.82, 42.43, 53.03, 63.64]
    elif num_broadcasters == 5:
        sniffer_distances_to_test = [12.5, 25.0, 37.5, 50.0, 62.5, 75.0]
    elif num_broadcasters == 6:
        sniffer_distances_to_test = [15.0, 30.0, 45.0, 60.0, 75.0, 90.0]

    # Initialize an empty list to store dictionaries
    sniffer_distance_list = []

    # Populate the list with dictionaries
    for sniffer_distance in sniffer_distances_to_test:
        sniffer_distance_list.append({
            'Broadcasters': num_broadcasters,
            'Sniffer_Distance': sniffer_distance,
            'RSSI': average_rssi[num_broadcasters].get(sniffer_distance, np.nan)
        })

    # Create a DataFrame from the list of dictionaries
    sniffer_distance_df = pd.DataFrame(sniffer_distance_list)
    print(sniffer_distance_df)
# PART 3
# Predict probabilities using the logistic regression model
data['Probability'] = clf.predict_proba(data[['Broadcasters', 'Sniffer_Distance', 'RSSI']])[:, 1]

# Calculate AUC
roc_auc = metrics.roc_auc_score(y_test, preds)
print("AUC:", roc_auc)

# Find threshold that balances sensitivity and specificity
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = threshold[optimal_idx]

# Use the optimal threshold
data['Predicted_Label'] = (data['Probability'] > optimal_threshold).astype(int)

print(optimal_threshold)
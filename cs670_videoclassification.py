import numpy as np
import os

video_features_path = '/CS670_Project/model/video_features.npy'
video_labels_path='/CS670_Project/model/video_labels.npy'
X = np.load(video_features_path)
y=np.load(video_labels_path)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Flatten the image data for PCA
X_flatten = X.reshape(X.shape[0], -1)

# Initialize PCA with the desired number of components
num_components = 2  # Adjust this as needed
pca = PCA(n_components=num_components)

# Fit PCA on the training data
pca.fit(X_flatten)

# Transform the training and testing data using the trained PCA
X_train_pca = pca.transform(X_flatten)

print(X_train_pca.shape)

plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Average Frame PCA Plot (2D)')
plt.colorbar(label='Class')
plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from sklearn.model_selection import train_test_split

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)  # Take the last time step output
        return torch.softmax(output, dim=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train)  
y_train_tensor = torch.tensor(y_train)
train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = LSTMClassifier(input_dim=2048, hidden_dim=128, num_layers=1, output_dim=11)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

X_test_tensor = torch.tensor(X_test)  # Features tensor
y_test_tensor = torch.tensor(y_test)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

all_predictions = []
all_probabilities = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
        all_predictions.extend(predicted.tolist())
        all_probabilities.extend(probabilities.tolist())

all_predictions = torch.tensor(all_predictions)
all_probabilities = torch.tensor(all_probabilities)

print("Predicted labels for test sequences:", all_predictions)
print("Class probabilities for test sequences:", all_probabilities)

torch.save(model.state_dict(), '/CS670_Project/lstm_model.pth')

model1 = LSTMClassifier(input_dim=2048, hidden_dim=128, num_layers=1, output_dim=11)

model1.load_state_dict(torch.load('/CS670_Project/lstm_model_pca.pth'))

model1.eval()
torch.save(model1, '/CS670_Project/lstm_model_fullpca.pth')

torch.save(model1, '/CS670_Project/lstm_model_full.pth')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test_tensor = torch.tensor(X_test)  # Features tensor
y_test_tensor = torch.tensor(y_test)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

all_predictions = []
all_probabilities = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model1(inputs)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
        all_predictions.extend(predicted.tolist())
        all_probabilities.extend(probabilities.tolist())

all_predictions = torch.tensor(all_predictions)
all_probabilities = torch.tensor(all_probabilities)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, all_predictions)
print("Accuracy:", accuracy)

"""
**LSTM 2 layers**"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from sklearn.model_selection import train_test_split

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)  # Add second LSTM layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        print(lstm_out.shape)
        lstm_out, _ = self.lstm2(lstm_out)  # Pass output of first LSTM to second LSTM
        print(lstm_out.shape)
        output = self.fc(lstm_out)   # Take the last time step output
        print(output.shape)
        return torch.softmax(output, dim=1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Prepare data loaders
X_train_tensor = torch.tensor(X_train)  # Features tensor
y_train_tensor = torch.tensor(y_train)
train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define model, loss function, and optimizer
model = LSTMClassifier(input_dim=2048, hidden_dim=128, num_layers=1, output_dim=11)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Evaluation
X_test_tensor = torch.tensor(X_test)  # Features tensor
y_test_tensor = torch.tensor(y_test)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

all_predictions = []
all_probabilities = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
        all_predictions.extend(predicted.tolist())
        all_probabilities.extend(probabilities.tolist())

all_predictions = torch.tensor(all_predictions)
all_probabilities = torch.tensor(all_probabilities)

print("Predicted labels for test sequences:", all_predictions)
print("Class probabilities for test sequences:", all_probabilities)

torch.save(model.state_dict(), '/CS670_Project/lstm2_model.pth')

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, all_predictions)
print("Accuracy:", accuracy)

"""**PCA**"""

model1 = LSTMClassifier(input_dim=2048, hidden_dim=128, num_layers=1, output_dim=11)

# Load the saved state dictionary into the model
model1.load_state_dict(torch.load('/CS670_Project/lstm_model_pca.pth'))

# Ensure the model is in evaluation mode
model1.eval()

model1

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib

# Flatten the image data for PCA
X_flatten = X.reshape(X.shape[0], -1)

# Initialize PCA with the desired number of components
num_components = 1000  # Adjust this as needed
pca = PCA(n_components=num_components)

# Fit PCA on the training data
pca.fit(X_flatten)

# Transform the training and testing data using the trained PCA
X_pca = pca.transform(X_flatten)
#joblib.dump(pca, '/CS670_Project/pca_model.pkl')

np.save("/CS670_Project/video_feats_pca",X_pca)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from sklearn.model_selection import train_test_split
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Prepare data loaders
X_train_tensor = torch.tensor(X_train)  # Features tensor
y_train_tensor = torch.tensor(y_train)
train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define model, loss function, and optimizer
model = LSTMClassifier(input_dim=1000, hidden_dim=128, num_layers=1, output_dim=11)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == targets).sum().item()
        total_samples += targets.size(0)

    # Calculate average loss and accuracy for the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct_predictions / total_samples

    # Print training statistics for the epoch
    print(f'Epoch [{epoch+1}/{20}], Loss: {loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

# Evaluation
X_test_tensor = torch.tensor(X_test)  # Features tensor
y_test_tensor = torch.tensor(y_test)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

all_predictions = []
all_probabilities = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
        all_predictions.extend(predicted.tolist())
        all_probabilities.extend(probabilities.tolist())

all_predictions = torch.tensor(all_predictions)
all_probabilities = torch.tensor(all_probabilities)

print("Predicted labels for test sequences:", all_predictions)
print("Class probabilities for test sequences:", all_probabilities)

torch.save(model.state_dict(), '/CS670_Project/lstm_model_pca.pth')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from sklearn.model_selection import train_test_split
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Prepare data loaders
X_train_tensor = torch.tensor(X_train)  # Features tensor
y_train_tensor = torch.tensor(y_train)
train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Define model, loss function, and optimizer
import matplotlib.pyplot as plt

model = LSTMClassifier(input_dim=1000, hidden_dim=512, num_layers=1, output_dim=11)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_losses,train_acc=[],[]
# Training loop
for epoch in range(10):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == targets).sum().item()
        total_samples += targets.size(0)
        running_loss += loss.item() * inputs.size(0)

    # Calculate average loss and accuracy for the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct_predictions / total_samples

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_correct_predictions = 0
    val_total_samples = 0
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_targets).item()
            _, val_predicted = torch.max(val_outputs, 1)
            val_correct_predictions += (val_predicted == val_targets).sum().item()
            val_total_samples += val_targets.size(0)

    # Calculate average validation loss and accuracy
    val_epoch_loss = val_loss / len(val_loader.dataset)
    val_epoch_accuracy = val_correct_predictions / val_total_samples

    # Print training and validation statistics for the epoch
    print(f'Epoch [{epoch+1}/{10}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.4f}')
    train_losses.append(epoch_loss)
    train_acc.append(epoch_accuracy)

# Plot loss curves
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Evaluation
X_test_tensor = torch.tensor(X_test)  # Features tensor
y_test_tensor = torch.tensor(y_test)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

all_predictions = []
all_probabilities = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
        all_predictions.extend(predicted.tolist())
        all_probabilities.extend(probabilities.tolist())

all_predictions = torch.tensor(all_predictions)
all_probabilities = torch.tensor(all_probabilities)

print("Predicted labels for test sequences:", all_predictions)
print("Class probabilities for test sequences:", all_probabilities)

#torch.save(model.state_dict(), '/CS670_Project/lstm_model_pca.pth')

train_acc

tr=[0.7208654558352546,
 0.8360474456742857,
 0.8821995263778336,
 0.904618337732247,0.90781,0.91005,0.9129,0.9153,0.91997]

plt.plot(train_losses, label='Training Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss trend')
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, all_predictions)
print("Accuracy:", accuracy)

accuracy = accuracy_score(y_test, all_predictions)
accuracy

model1 = LSTMClassifier(input_dim=2048, hidden_dim=128, num_layers=1, output_dim=11)
model1.load_state_dict(torch.load('/CS670_Project/lstm_model_pca.pth'))
model1.eval()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test_tensor = torch.tensor(X_test)  # Features tensor
y_test_tensor = torch.tensor(y_test)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

all_predictions = []
all_probabilities = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model1(inputs)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
        all_predictions.extend(predicted.tolist())
        all_probabilities.extend(probabilities.tolist())

all_predictions = torch.tensor(all_predictions)
all_probabilities = torch.tensor(all_probabilities)

y_test, all_predictions

from sklearn.metrics import precision_score,recall_score,f1_score
precision = precision_score(y_test, all_predictions, average="weighted")
recall = recall_score(y_test, all_predictions, average="weighted")
f1 = f1_score(y_test, all_predictions, average="weighted")
precision,recall,f1

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming y_true contains the true labels and y_pred contains the predicted labels
# Replace y_true and y_pred with your actual true and predicted labels
# Make sure both y_true and y_pred are 1-dimensional arrays/lists of the same length

# Calculate the confusion matrix
cm = confusion_matrix(y_test, all_predictions)

# Display the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Video Confusion Matrix')
plt.show()

"""**Neural**"""

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform PCA
pca = PCA(n_components=1000)
X_flatten = X_train.reshape(X_train.shape[0], -1)
pca.fit(X_flatten)
X_train_pca = pca.transform(X_flatten)
X_test_pca = pca.transform(X_test.reshape(X_test.shape[0], -1))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_pca, y_train, test_size=0.2, random_state=42)

# Define a simple feedforward neural network classifier
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_pca.shape[1],)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(11, activation='softmax')  # 11 is the number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test_pca, y_test)
print('Test accuracy:', test_acc)

"""**Video Middle Frame**"""

import numpy as np
import os

thirdframe_features_path = '/CS670_Project/model/thirdframe_features.npy'
thirdframe_labels_path='/CS670_Project/model/thirdframe_labels.npy'
X_tf = np.load(thirdframe_features_path)
y_tf=np.load(thirdframe_labels_path)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Flatten the image data for PCA
X_flatten = X_tf.reshape(X_tf.shape[0], -1)

# Initialize PCA with the desired number of components
num_components = 2  # Adjust this as needed
pca = PCA(n_components=num_components)

# Fit PCA on the training data
pca.fit(X_flatten)

# Transform the training and testing data using the trained PCA
X_train_pca = pca.transform(X_flatten)

print(X_train_pca.shape)

plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_tf, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Plot (2D)')
plt.colorbar(label='Class')
plt.show()

X_tf=X_tf.reshape(-1,2048)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from sklearn.model_selection import train_test_split

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)  # Take the last time step output
        return torch.softmax(output, dim=1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tf, y_tf, test_size=0.2, random_state=42)
# Prepare data loaders
X_train_tensor = torch.tensor(X_train)  # Features tensor
y_train_tensor = torch.tensor(y_train)
train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define model, loss function, and optimizer
model = LSTMClassifier(input_dim=2048, hidden_dim=128, num_layers=1, output_dim=11)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Evaluation
X_test_tensor = torch.tensor(X_test)  # Features tensor
y_test_tensor = torch.tensor(y_test)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

all_predictions = []
all_probabilities = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
        all_predictions.extend(predicted.tolist())
        all_probabilities.extend(probabilities.tolist())

all_predictions = torch.tensor(all_predictions)
all_probabilities = torch.tensor(all_probabilities)

print("Predicted labels for test sequences:", all_predictions)
print("Class probabilities for test sequences:", all_probabilities)

torch.save(model.state_dict(), '/CS670_Project/middle_frame_lstm_model.pth')

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, all_predictions)
print("Accuracy:", accuracy)


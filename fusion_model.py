import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from keras.models import load_model
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

X_audio2=np.load('/CS670_Project/final_audio_feat2.npy')
X_video2=np.load('/CS670_Project/final_video_feat2.npy')
labels=np.load('/CS670_Project/final_labels2.npy')

X_video2.shape,X_audio2.shape,labels.shape

num_samples = len(X_audio2)
test_size = 0.2  

test_indices = np.random.choice(num_samples, size=int(test_size * num_samples), replace=False)

# Select data for the test set using the random indices
X_test_audio = X_audio2[test_indices]
X_test_video = X_video2[test_indices]
Y_test = labels[test_indices]
train_indices = [i for i in np.arange(num_samples) if i not in test_indices]

X_train_audio = X_audio2[train_indices]
X_train_video = X_video2[train_indices]
y_train = labels[train_indices]

X_train_video.shape, X_test_video.shape

"""# Probabilities Experiments"""

from sklearn.model_selection import train_test_split

class LSTMClassifier(nn.Module):
    def _init_(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMClassifier, self)._init_()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)  
        return torch.softmax(output,dim=1)

"""### **Combined Probabilities**"""

classifier_audio = load_model('/models/audiomodel.h5')
classifier_video = torch.load('/models/lstm_model_full.pth')
X_test_video_tensor = torch.tensor(X_test_video, dtype=torch.float32)

classifier_video.eval()

with torch.no_grad():
    y_pred_video = classifier_video(X_test_video_tensor)
y_pred_audio = classifier_audio.predict(X_test_audio)
n_classes=11
accuracy_audio = accuracy_score(Y_test, np.argmax(y_pred_audio, axis=1))
accuracy_video = accuracy_score(Y_test, np.argmax(y_pred_video.numpy(), axis=1))

prob_audio = accuracy_audio / (accuracy_audio + accuracy_video)
prob_video = accuracy_video / (accuracy_audio + accuracy_video)

y_pred_combined_audio = []
y_pred_combined_video = []

for pred_audio, pred_video in zip(y_pred_audio, y_pred_video):
    combined_probabilities_audio = np.zeros(n_classes)
    combined_probabilities_video = np.zeros(n_classes)

    combined_probabilities_audio[np.argmax(pred_audio)] += prob_audio
    combined_probabilities_video[np.argmax(pred_video)] += prob_video

    y_pred_combined_audio.append(np.argmax(combined_probabilities_audio))
    y_pred_combined_video.append(np.argmax(combined_probabilities_video))

y_pred_combined = []

for pred_audio, pred_video in zip(y_pred_combined_audio, y_pred_combined_video):
    combined_probabilities = np.zeros(n_classes)

    combined_probabilities[pred_audio] += prob_audio
    combined_probabilities[pred_video] += prob_video

    y_pred_combined.append(np.argmax(combined_probabilities))

accuracy_combined = accuracy_score(Y_test, y_pred_combined)
print("Accuracy of combined predictions:", accuracy_combined)

print("Accuracy of audio model:", accuracy_audio)
print("Accuracy of video model:", accuracy_video)

"""### **Naive Bayes**"""

X_audio3=X_audio2.reshape(-1,5*40)
X_combined = np.concatenate((X_video2, X_audio3), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_combined, labels, test_size=0.2, random_state=42)

test_accuracies = []
train_accuracies = []

for i in range(10):
    nb_classifier = GaussianNB()

    nb_classifier.fit(X_train, y_train)

    y_pred_test = nb_classifier.predict(X_test)
    y_pred_train = nb_classifier.predict(X_train)

    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)

    test_accuracies.append(test_accuracy)
    print(test_accuracy)
    train_accuracies.append(train_accuracy)
    print(train_accuracy)
    

print("Average Train Accuracy:", sum(train_accuracies) / len(train_accuracies))
print("Average Test Accuracy:", sum(test_accuracies) / len(test_accuracies))

precision = precision_score(Y_test, y_pred_test, average="weighted")
recall = recall_score(Y_test, y_pred_test, average="weighted")
f1 = f1_score(Y_test, y_pred_test, average="weighted")
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)



"""# Decision Tree"""


classifier_audio = load_model('/CS670_Project/audio/audiomodel.h5')
classifier_video = torch.load('/CS670_Project/lstm_model_fullpca.pth')

#X_train_audio3=X_train_audio.reshape(-1,5*40)

X_test_video_tensor = torch.tensor(X_test_video, dtype=torch.float32)

classifier_video.eval()

audio_probs = classifier_audio.predict(X_train_audio)
X_tr_video_tensor = torch.tensor(X_train_video, dtype=torch.float32)
video_probs = classifier_video(X_tr_video_tensor)

combined_probs = np.concatenate((audio_probs, video_probs.detach().numpy()), axis=1)

labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8 ,9 ,10])

clf = DecisionTreeClassifier()

clf.fit(combined_probs, y_train)
with torch.no_grad():
    y_pred_video = classifier_video(X_test_video_tensor)
y_pred_audio = classifier_audio.predict(X_test_audio)
X_test_combined = np.concatenate((y_pred_audio, y_pred_video), axis=1)

y_pred = clf.predict(X_test_combined)

accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)
precision = precision_score(Y_test, y_pred, average="weighted")
recall = recall_score(Y_test, y_pred, average="weighted")
f1 = f1_score(Y_test, y_pred, average="weighted")
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)


"""# Gradient Boosting"""


classifier_audio = load_model('/CS670_Project/audio/audiomodel.h5')
classifier_video = torch.load('/CS670_Project/lstm_model_fullpca.pth')

"""## Training"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np


classifier_audio = load_model('/CS670_Project/audio/audiomodel.h5')
classifier_video = torch.load('/CS670_Project/lstm_model_fullpca.pth')

X_test_video_tensor = torch.tensor(X_test_video, dtype=torch.float32)

classifier_video.eval()

audio_probs = classifier_audio.predict(X_train_audio)
X_tr_video_tensor = torch.tensor(X_train_video, dtype=torch.float32)
video_probs = classifier_video(X_tr_video_tensor)

combined_probs = np.concatenate((audio_probs, video_probs.detach().numpy()), axis=1)

gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(combined_probs, y_train)
with torch.no_grad():
    y_pred_video = classifier_video(X_test_video_tensor)
y_pred_audio = classifier_audio.predict(X_test_audio)
X_test_combined = np.concatenate((y_pred_audio, y_pred_video), axis=1)

y_pred = gb_classifier.predict(X_test_combined)

accuracy = accuracy_score(Y_test, y_pred)
accuracy_audio = accuracy_score(Y_test, np.argmax(y_pred_audio, axis=1))
accuracy_video = accuracy_score(Y_test, np.argmax(y_pred_video.numpy(), axis=1))

print("Audio Accuracy:", accuracy_audio)
print("Video Accuracy:", accuracy_video)
print("Accuracy:", accuracy)

"""## Gradient Boosting Testing"""

import joblib
model_path = '/CS670_Project/gb_classifier_model2.pkl'  

gb_classifier = joblib.load(model_path)

X_test_video_tensor = torch.tensor(X_test_video, dtype=torch.float32)

with torch.no_grad():
    y_pred_video = classifier_video(X_test_video_tensor)
y_pred_audio = classifier_audio.predict(X_test_audio)
X_test_combined = np.concatenate((y_pred_audio, y_pred_video), axis=1)

y_pred = gb_classifier.predict(X_test_combined)

accuracy = accuracy_score(Y_test, y_pred)
accuracy_audio = accuracy_score(Y_test, np.argmax(y_pred_audio, axis=1))
accuracy_video = accuracy_score(Y_test, np.argmax(y_pred_video.numpy(), axis=1))

print("Audio Accuracy:", accuracy_audio)
print("Video Accuracy:", accuracy_video)
print("Accuracy:", accuracy)


cm = confusion_matrix(Y_test, np.argmax(y_pred_video.numpy(), axis=1))

mapping_dict = {
    0: "advertisement",
    1: "drama",
    2: "entertainment",
    3: "interview",
    4: "live_broadcast",
    5: "movie",
    6: "play",
    7: "recitation",
    8: "singing",
    9: "speech",
    10: "vlog"
    }
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=mapping_dict.values(), yticklabels=mapping_dict.values())  # Use mapping_dict for labels
plt.title('Video Model Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

precision = precision_score(Y_test, y_pred, average="weighted")
recall = recall_score(Y_test, y_pred, average="weighted")
f1 = f1_score(Y_test, y_pred, average="weighted")
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

precision = precision_score(Y_test,  np.argmax(y_pred_audio, axis=1), average="weighted")
recall = recall_score(Y_test,  np.argmax(y_pred_audio, axis=1), average="weighted")
f1 = f1_score(Y_test,  np.argmax(y_pred_audio, axis=1), average="weighted")
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

precision = precision_score(Y_test,  np.argmax(y_pred_video.numpy(), axis=1), average="weighted")
recall = recall_score(Y_test,  np.argmax(y_pred_video.numpy(), axis=1), average="weighted")
f1 = f1_score(Y_test,  np.argmax(y_pred_video.numpy(), axis=1), average="weighted")
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

##Training
y_trpred = gb_classifier.predict(combined_probs)

# Evaluate the predictions
# Replace y_test with your test labels
accuracy = accuracy_score(y_train, y_trpred)
accuracy_audio = accuracy_score(y_train, np.argmax(audio_probs, axis=1))
accuracy_video = accuracy_score(y_train, np.argmax(video_probs.detach().numpy(), axis=1))
print("Audio Accuracy:", accuracy_audio)
print("Video Accuracy:", accuracy_video)
print("Accuracy:", accuracy)

joblib.dump(gb_classifier, '/CS670_Project/gb_classifier_model2.pkl')


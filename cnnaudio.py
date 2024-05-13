import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
#for loading and visualizing audio files
import librosa
import librosa.display
import numpy as np
#to play audio
import math
from keras.models import load_model

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



SAMPLE_RATE = 22050
TRACK_DURATION = 5 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def features_extractor(file, n_fft=2048, hop_length=512, num_segments=5):
  samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
  features=[]
  for d in range(num_segments):
    start = samples_per_segment * d
    finish = start + samples_per_segment
    audio, sample_rate = librosa.load(file,sr=SAMPLE_RATE)
    mfccs_features = librosa.feature.mfcc(y=audio[start:finish], sr=sample_rate, n_mfcc=40, n_fft=n_fft, hop_length=hop_length)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    features.append(mfccs_scaled_features)
  return features


def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


#declare a path to batch file for loop for folder in 



def load_data(directory):
    extracted_features=[]
    labels = []
    file_name=[]
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            file_path = os.path.join(directory, file)
            genre = file.split('-')[1]
            labels.append(genre)
            file_name.append(file)
            features = features_extractor(file_path)
            extracted_features.append(features)
            print(file_path+"is done")
    return np.array(extracted_features), np.array(labels),np.array(file_name)

  # Load your labeled dataset

def prepare_datasets(test_size, validation_size):
 # load data
    
    X_feature = np.load('C:/Users/harsh/source/saved/audio_features.npy')
    strings_array = np.load('C:/Users/harsh/source/saved/labels.npy')
    ''' path='data'
    X_feature=[]
    Y_feature=[]
    for folder in os.listdir(path):
        directory=os.path.join(path,folder)
        X, y,z = load_data(directory)
        names_array=z
        for j in range(len(X)):
            X_feature.append(X[j])
            Y_feature.append(y[j])
    
    np.save('C:/Users/harsh/source/saved/audio_features.npy', X_feature)
    np.save('C:/Users/harsh/source/saved/labels.npy', Y_feature)
    np.save('C:/Users/harsh/source/saved/names.npy',names_array)'''
    mapping_dict = {"advertisement":0, 
        "drama":1, 
        "entertainment": 2, 
        "interview": 3, 
        "live_broadcast": 4, 
        "movie": 5,
        "play": 6,
        "recitation": 7,
        "singing": 8,
        "speech": 9,
        "vlog": 10}
    Y_feature = np.array([mapping_dict.get(string, "default_value") for string in strings_array])
    # Flatten the image data for PCA
    X_flatten = X_feature.reshape(X_feature.shape[0], -1)

    
    X_train, X_test, y_train, y_test = train_test_split(X_feature, Y_feature, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    X_train = X_train.reshape(X_train.shape[0], 5, 40, 1)
    X_validation = X_validation.reshape(X_validation.shape[0], 5, 40,1)
    X_test = X_test.reshape(X_test.shape[0], 5, 40,1)

    return X_train, X_validation, X_test, y_train, y_validation, y_test

#n ,5,40,1
#n,11

def build_model(input_shape):
    # build network topology
    model = keras.Sequential()
    # n-f+2p/s +1

    # 1st conv layer
    model.add(keras.layers.Conv2D(2048, (1, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(1, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(1024, (1, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(1, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 4rth conv layer
    model.add(keras.layers.Conv2D(256, (1, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(1, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(11, activation='softmax'))

    return model


def predict(model, X, y):

    X = X[np.newaxis, ...] 
    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))


if __name__ == "__main__":

    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)

    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    ''''''
    model.summary()
    print(X_train.dtype)
    print(y_train.dtype)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=15, callbacks=[early_stopping])
    plot_history(history)
    '''X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)
    model=load_model('C:/Users/harsh/source/model/audiomodel.h5')'''
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    # Save the model
    # model.save('C:/Users/harsh/source/model/audiomodel.h5')
    print('\nTest accuracy:', test_acc)

    y_pred_prob = model.predict(X_test) 
    y_pred = np.argmax(y_pred_prob, axis=1)  
    y_true = y_test 
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix using seaborn heatmap
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
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=mapping_dict.values(), yticklabels=mapping_dict.values())  # Use mapping_dict for labels
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()
    # predict sample
    #predict(model, X_to_predict, y_to_predict)
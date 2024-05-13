
import cv2
import numpy as np
import os
import tensorflow
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model

base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(frame):
    frame = cv2.resize(frame, (299, 299))  # Resize frame to match input size of InceptionV3
    frame = preprocess_input(frame)  # Preprocess frame
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    features = model.predict(frame)  # Extract features using InceptionV3
    return features

def compute_average_frame_and_features(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frame = np.zeros((299, 299, 3), dtype=np.float32)  # Initialize total frame

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (299, 299))

        frame = frame.astype(np.float32)
        total_frame += frame

        frame_count += 1
    cap.release()

    if frame_count == 0:
        return None  # Return None if no frames were read

    average_frame = (total_frame / frame_count).astype(np.uint8)

    average_frame_features = extract_features(average_frame)

    return average_frame_features


folder_path = "/content/drive/MyDrive/CS670/train/video/"
video_folders = os.listdir(folder_path)

labels={"advertisement":0,
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




for i, video_folder in enumerate(video_folders):
    folder_full_path = os.path.join(folder_path, video_folder)
    files = os.listdir(folder_full_path)
    feat=[]
    names=[]
    y=[]
    val=video_folder.split('_')[1]

    for j, name in enumerate(files):
        print(name)
        video_path = os.path.join(folder_full_path, name)

        video_features = compute_average_frame_and_features(video_path)
        if video_features is not None:
            feat.append(video_features)
            names.append(name)
            y.append(labels[name.split("-")[1]])
            print("Shape of video features array in "+video_folder+" :", video_features.shape)
        else:
            print("No frames read from video:", name)

    feat=np.array(feat)
    names=np.array(names)
    y=np.array(y)
    np.save('/CS670_Project/video_features_'+str(val)+'.npy', feat)
    np.save('/CS670_Project/labels_'+str(val)+'.npy', y)
    np.save('/CS670_Project/names_'+str(val)+'.npy', names)




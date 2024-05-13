import numpy as np
audio_feat_path = '/CS670_Project/audio/audio_features.npy'
video_feat_path='/CS670_Project/video_feats_pca.npy'
audio_names_path = '/CS670_Project/audio/names_final.npy'
video_names_path='/CS670_Project/model/video_names.npy'
audio_labels_path = '/CS670_Project/audio/labels.npy'
video_labels_path = '/CS670_Project/model/video_labels.npy'

X_video = np.load(video_feat_path)
X_audio = np.load(audio_feat_path)
y=[]
video_names= np.load(video_names_path)
audio_names= np.load(audio_names_path)

audio_names.shape, video_names.shape
audio_labels= np.load(audio_labels_path)
video_labels= np.load(video_labels_path)

video_feats=[]
audio_feats=[]
labels=[]
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


for i,name in enumerate(video_names):
  n2=name.split('.')[0]+'.wav'
  print(i,name)
  if n2 in audio_names:
    index = np.where(audio_names == n2)[0][0]
    if mapping_dict[audio_labels[index]] == video_labels[i]:
      print("Appending..")
      video_feats.append(X_video[i])
      audio_feats.append(X_audio[index])
      labels.append(video_labels[i])

X_video2=np.array(video_feats)
X_audio2=np.array(audio_feats)
y=np.array(labels)

np.save('/CS670_Project/final_video_feat2',X_video2)
np.save('/CS670_Project/final_audio_feat2',X_audio2)
np.save('/CS670_Project/final_labels2',y)
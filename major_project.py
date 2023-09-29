# -*- coding: utf-8 -*-
"""Major_Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MWNWy0c1YubkgJUs26t4EX9hojN6__kL
"""

! pip install -q kaggle
from google.colab import files
files.upload()
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets list
!kaggle datasets download -d adityajn105/flickr8k

!unzip flickr8k.zip

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import Model,Input, layers,models
import os
from tqdm import tqdm

from tensorflow.keras.preprocessing import image,text

img_model = VGG16()
img_model = Model(inputs=img_model.inputs,outputs=img_model.layers[-2].output)

print(img_model.summary())

img_features = {}
img_dir = '/content/Images'
for img_name in tqdm(os.listdir(img_dir)):
  img_path = os.path.join(img_dir,img_name)
  img = image.load_img(img_path,target_size=(224,224))
  img = image.img_to_array(img)
  img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
  img = preprocess_input(img)
  features = img_model.predict(img,verbose=0)
  img_id = img_name.split('.')[0]
  img_features[img_id] = features

import pickle

with open('img_features.pkl','wb') as img_feat_file:
  pickle.dump(img_features,img_feat_file)

img_features = pickle.load(open('img_features.pkl','rb'))

with open('/content/captions.txt','r') as file:
  next(file)
  caption_txt = file.read()

image_caption_map = {}
for line in tqdm(caption_txt.split('\n')):
  tokens = line.split(',')
  if len(line) < 2:
    continue
  img_name,caption = tokens[0], tokens[1:]
  img_name = img_name.split('.')[0]
  caption = ' '.join(caption)
  if img_name not in image_caption_map:
    image_caption_map[img_name] = []
  image_caption_map[img_name].append(caption)

def clean_captions(image_caption_map):
  for img_id,captions in image_caption_map.items():
    for i in range(len(captions)):
      processed_caption = captions[i]
      processed_caption = processed_caption.lower()
      processed_caption = processed_caption.replace('[^a-z]','')
      processed_caption = processed_caption.replace('\s+','')
      processed_caption = processed_caption.replace('"','')
      processed_caption = '<start> '+ ' '.join([char for char in processed_caption.split() if len(char)>1])+' <end>'
      captions[i] = processed_caption

clean_captions(image_caption_map)

with open('img_caption_map.pkl','wb') as img_cap_file:
  pickle.dump(image_caption_map,img_cap_file)

image_caption_map = pickle.load(open('img_caption_map.pkl','rb'))

image_caption_map['2073964624_52da3a0fc4']

all_captions = []
for img_id,captions in image_caption_map.items():
  all_captions.extend(captions)

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index)+1

print(vocab_size)

max_caption_len = max(len(caption.split()) for caption in all_captions)

print(max_caption_len)

img_ids = list(image_caption_map.keys())
split = int(len(img_ids)*0.90)
train = img_ids[:split]
test = img_ids[split:]

print(len(train))
print(len(test))

import numpy as np
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

def data_generator(data_keys,image_caption_map,img_features,tokenizer,max_caption_len,vocab_size,batch_size):
  X1 , X2, y = list(), list(), list()
  size = 0
  while True:
    for key in data_keys:
      size+=1
      captions = image_caption_map[key]
      for caption in captions:
        seq = tokenizer.texts_to_sequences([caption])[0]
        for i in range(1,len(seq)):
          in_seq, out_seq = seq[:1],seq[i]
          in_seq = pad_sequences([in_seq],maxlen=max_caption_len)[0]
          out_seq = to_categorical([out_seq],num_classes=vocab_size)[0]
          X1.append(img_features[key][0])
          X2.append(in_seq)
          y.append(out_seq)
    if size==batch_size:
      X1 , X2, y = np.array(X1), np.array(X2), np.array(y)
      yield [X1,X2], y
      X1 , X2, y = list(), list(), list()
      size = 0

inputs1 = Input(shape=(4096,))
fe1 = layers.Dropout(0.4)(inputs1)
fe2 = layers.Dense(256,activation='relu')(fe1)

inputs2 = Input(shape=(max_caption_len))
se1 = layers.Embedding(vocab_size,256,mask_zero=True)(inputs2)
se2 = layers.Dropout(0.4)(se1)
se3 = layers.LSTM(256)(se2)

decoder1 = layers.add([fe2,se3])
decoder2 = layers.Dense(256,activation='relu')(decoder1)
outputs = layers.Dense(vocab_size,activation='softmax')(decoder2)

endec_model = Model(inputs=[inputs1,inputs2],outputs=outputs)
endec_model.compile(loss='categorical_crossentropy',optimizer='adam')

plot_model(endec_model,show_shapes=True)

epochs = 25
batch_size = 2
step = len(train)//batch_size
for i in range(epochs):
  generator = data_generator(train,image_caption_map,img_features,tokenizer,max_caption_len,vocab_size,batch_size)
  endec_model.fit(generator,epochs=1,steps_per_epoch=step,verbose=1)
import numpy as np
import json
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# تحميل بيانات التدريب
with open('/home/abdulwahab/Downloads/abdelwhab/abdelwhab/Chatbot/Chatbot/intents.json') as file:
    data = json.load(file)

# تحضير البيانات
patterns = []
responses = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['tag'])
    labels.append(intent['tag'])

# تحويل النصوص إلى تسلسلات أرقام
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
X = tokenizer.texts_to_sequences(patterns)
X = pad_sequences(X, maxlen=20)  # افترض أن الطول الأقصى هو 20

# ترميز التسميات
le = LabelEncoder()
y = le.fit_transform(responses)
y = to_categorical(y)

# بناء نموذج بسيط
model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# تدريب النموذج
model.fit(X, y, epochs=10, batch_size=8, verbose=1)

# حفظ النموذج والموارد
model.save('/home/abdulwahab/Downloads/abdelwhab/abdelwhab/Chatbot/Chatbot/SaveModels/model.h5')

with open('/home/abdulwahab/Downloads/abdelwhab/abdelwhab/Chatbot/Chatbot/SaveModels/toknizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('/home/abdulwahab/Downloads/abdelwhab/abdelwhab/Chatbot/Chatbot/SaveModels/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

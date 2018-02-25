from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pprint as pp


X_train = np.array([[float(j) for j in i.rstrip().split(",")]
                    for i in open("train.csv").readlines()])
Y_train = X_train[:,-1]
X_train = X_train[:,0:-1]

X_test = np.array([[float(j) for j in i.rstrip().split(",")]
                    for i in open("test.csv").readlines()])
Y_test = X_test[:,-1]
X_test = X_test[:,0:-1]

inputs = Input(shape=[9])

x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)

probs = Dense(1, activation='sigmoid')(x)


model1 = Model(inputs=inputs, outputs=probs)
model1.compile(optimizer='sgd', loss='binary_crossentropy')

model1.fit(X_train, Y_train, epochs=100, batch_size=128,
           validation_split=0.15, verbose=True)

# save model
with open('model.json', "w") as model_save_file:
    model_json = model1.to_json()
    model_save_file.write(model_json)

# weights:
weights = model1.get_weights() # as numpy arrays
pp.pprint(weights)
for w in weights:
   print("w.shape=", w.shape)

np.savetxt('w1.csv', weights[0], delimiter=",")
np.savetxt('b1.csv', weights[1], delimiter=",")

np.savetxt('w2.csv', weights[2], delimiter=",")
np.savetxt('b2.csv', weights[3], delimiter=",")

np.savetxt('w3.csv', weights[4], delimiter=",")
np.savetxt('b3.csv', weights[5], delimiter=",")
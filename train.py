import argparse
from scrape import one_hot_encoded,process,encode_labels
import os
import numpy as np
#from sklearn.externals import joblib
import joblib
import pickle
import pandas as pd
from azureml.core import  Dataset
from azureml.core import Run
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import tensorflow as tf


maxlen=32
N_LANG=10
parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--test-size',type=float)
parser.add_argument('--first-layer-neurons', type=int)
parser.add_argument('--second-layer-neurons', type=int)
parser.add_argument('--third-layer-neurons', type=int)
parser.add_argument('--fourth-layer-neurons', type=int)
parser.add_argument('--learning-rate',type=float)
parser.add_argument('--dropout1',type=float)
parser.add_argument('--dropout2',type=float)
parser.add_argument('--dropout3',type=float)
parser.add_argument('--weight-constraint',type=int)

#arser.add_argument('--regularization', type=float, dest='reg', default=0.01, help='regularization rate')
args = parser.parse_args()


def getmodel():
    model = Sequential()
    model.add(Flatten())
    if(args.weight_constraint!=0):
        model.add(Dense(args.first_layer_neurons,input_dim=23*maxlen,activation='sigmoid',
                    kernel_constraint=maxnorm(args.weight_constraint)))
    else:
        model.add(Dense(args.first_layer_neurons,input_dim=23*maxlen,activation='sigmoid'))
        
    model.add(Dense(args.second_layer_neurons, activation='sigmoid'))
    if(args.dropout1!=0):
        model.add(Dropout(args.dropout1))
    if(args.third_layer_neurons!=0):
        model.add(Dense(args.third_layer_neurons, activation='sigmoid'))
    if(args.dropout2!=0):
        model.add(Dropout(args.dropout2))
    if(args.fourth_layer_neurons!=0):
        model.add(Dense(args.fourth_layer_neurons, activation='sigmoid'))
    if(args.dropout3!=0):
        model.add(Dropout(args.dropout3))
    model.add(Dense(N_LANG, activation='softmax'))
    opt = Adam(lr=args.learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def load_data():
    
    ws = run.experiment.workspace
    dataset = Dataset.get_by_id(ws, id=args.input_data)
    df = dataset.to_pandas_dataframe()
    X=[]
    Y=[]
    for x,y in zip(df.words,df.lang):
        X.append(one_hot_encoded(process(x)))
        Y.append(encode_labels(y))
    return np.array(X),np.array(Y)

class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log('Loss', log['val_loss'])
        run.log('Accuracy', log['val_accuracy'])
# load the TabularDataset to pandas DataFrame


run = Run.get_context()
X,Y=load_data()

i=args.test_size
j=1.0-i
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=i, random_state=1,shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(i/j), random_state=1,shuffle=True)

model=getmodel()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0,patience=10)
callbacks_list = [LogRunMetrics(),es]
history=model.fit(X_train, y_train, epochs=1000, batch_size=args.batch_size, validation_data=(X_val, y_val), callbacks=callbacks_list,verbose=0,shuffle=True)
    #plot_history(history,'lrate:{} testsize:{}'.format(l,j))
    #print('scores:',model.metrics[0],model.metrics[1])
score=model.evaluate(X_test,y_test,verbose=0)
run.log("Final test loss", score[0])
print('Test loss:', score[0])

run.log('Final test accuracy', score[1])
print('Test accuracy:', score[1])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
x = range(1, len(acc) + 1)

plt.figure(figsize=(6, 3))
plt.title('Best Model Accuracy', fontsize=14)
plt.plot(x, acc, 'b-', label='Training acc')
plt.plot(x, val_acc, 'r--', label='Validation acc')
plt.legend(fontsize=12)
plt.grid(True)
# log an image
run.log_image('Train vs Val - Accuracy ', plot=plt)

plt.figure(figsize=(6, 3))
plt.title('Best Model Loss', fontsize=14)
plt.plot(x, loss, 'b', label='Training loss')
plt.plot(x, val_loss, 'r', label='Validation loss')
plt.legend(fontsize=12)
plt.grid(True)
run.log_image('Train vs Val - Loss ', plot=plt)

os.makedirs('./outputs/model', exist_ok=True)
print('Export the model to model.pkl')
f = open('./outputs/model/model.pkl', 'wb')
pickle.dump(model, f)
f.close()

model.save_weights('./outputs/model/model.h5')
os.makedirs('./outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=model, filename='./outputs/bestmodel.pkl')

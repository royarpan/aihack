from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import cv2             #opencv library used to extract color data from image
import numpy as np     #used for arrays
from keras.models import load_model, Model
from keras.utils import to_categorical
from keras import backend as K
import tensorflow as tf 
import h5py

app = Flask(__name__)
app.config['TEST_FOLDER'] = 'uploads'
app.config['TRAIN_FOLDER'] = 'genuine'
userfilemap='userfilemap.txt'


@app.route('/train')
def train():
    return render_template('trainform.html')

@app.route('/submittedtrain', methods=['POST'])
def submittedtrain_form():
    name = request.form['name']
    email = request.form['email']
    comments = request.form['comments']
    f = request.files['file']
    filename=secure_filename(f.filename)
    n=len(filename)
    filename=filename[0:n-3]+filename[-3:].lower()
    print filename
    with open(userfilemap, 'a') as the_file:
        the_file.write("\n"+name.replace(" ","")+" "+filename)
    f.save(os.path.join(app.config['TRAIN_FOLDER'],filename))
    
    f = open(userfilemap)
    setofusers=set()
    userdict={}
    y=[]    #Y component of dataset
    genfilelist=[]
    line=f.readline()
    numfiles=0
    while line:
        numfiles=numfiles+1
        username = line.split()[0]
        filename = line.split()[1]
        genfilelist.append(filename)
        if username not in setofusers:
            userdict[username]=len(setofusers)
            setofusers.add(username)       
        y.append(userdict[username])
        line = f.readline()

    print("No. of users= ",len(setofusers))
    print ("User dictionary: ",userdict)
    Ygen=to_categorical(y,len(setofusers))   

    if(comments=='Add file'):
        return render_template(
        'submitted_train.html',
        name=name,
        email=email,
        filename=filename,
        comments=comments)
    else:
        gen=[]  #X component of dataset
        for i in range(numfiles):
            filepath=os.path.join(app.config['TRAIN_FOLDER'],genfilelist[i])
            print ("Filepath ",i," : ",filepath)
            img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
            pixelmap=np.asarray(img)
            print pixelmap.shape
            pixelmap = cv2.resize(pixelmap,(250,150),interpolation=cv2.INTER_AREA)  #resize the image to manage the computational complexity in training
            pixelmap = np.expand_dims(pixelmap, axis=2) #to map it to ML model input
            pixelmap=pixelmap/255 #normalize the image data
            gen.append(np.asarray(pixelmap))

        hf = h5py.File('genuine/dataset_genuine.h5', 'w') #store the dataset locally
        hf.create_dataset('dataset_1', data=gen)
        hf.create_dataset('dataset_2', data=Ygen)
        hf.close()

        from keras.models import Sequential
        from keras.layers import Input, Add, Dense, Dropout, Flatten, Lambda, ELU, Activation, Conv2D, BatchNormalization, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
        from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D
        from keras.optimizers import SGD, Adam, RMSprop

        row, col, ch = 150, 250, 1     #image shape input to the 1st layer of the CNN
        num_classes=len(setofusers)    #number of people whose signatures have been sampled

        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(row, col, ch)))

        # CNN model - Building the model suggested in DeepWriter paper (handwriting recognition model)

        model.add(Convolution2D(filters= 32, kernel_size =(5,5), strides= (2,2), padding='same', name='conv1')) #96
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2), name='pool1'))

        model.add(Convolution2D(filters= 64, kernel_size =(3,3), strides= (1,1), padding='same', name='conv2'))  #256
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2), name='pool2'))

        model.add(Convolution2D(filters= 128, kernel_size =(3,3), strides= (1,1), padding='same', name='conv3'))  #256
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2), name='pool3'))


        model.add(Flatten())
        model.add(Dropout(0.5))

        model.add(Dense(1024, name='dense1')) 

        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(1024, name='dense2'))  
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(num_classes,name='output'))
        model.add(Activation('softmax'))  #softmax since output is within 50 classes

        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        #print(model.summary())

        model.fit(x=np.array(gen),y=np.array(Ygen),epochs=40,batch_size=5,verbose=1)
        
        model.save('deepwriter.h5')
        comments_response="Training completed successfully for full dataset"
        K.clear_session()

        return render_template(
        'submitted_train.html',
        name=name,
        email=email,
        filename=filename,
        comments=comments_response)


@app.route('/test')
def test():
    return render_template('testform.html')

@app.route('/submittedtest', methods=['POST'])
def submittedtest_form():
    name = request.form['name']
    email = request.form['email']
    comments = request.form['comments']
    f = request.files['file']
    filename=secure_filename(f.filename)
    uploadedfilepath=os.path.join(app.config['TEST_FOLDER'],filename)
    f.save(uploadedfilepath)
    #print uploadedfilepath
    img=cv2.imread(uploadedfilepath,cv2.IMREAD_GRAYSCALE)
    pixelmap=np.asarray(img)
    #print pixelmap.shape
    pixelmap=cv2.resize(pixelmap,(250,150),interpolation= cv2.INTER_AREA)
    pixelmap=np.expand_dims(pixelmap, axis=2)
    pixelmap=pixelmap/255  #normalize the data
    imgarr=np.array([pixelmap])
    global graph
    graph = tf.get_default_graph()
    with graph.as_default():
        model = load_model("deepwriter.h5")
        results=model.predict(imgarr)

    f = open(userfilemap)
    setofusers=set()
    userdict={}
    line=f.readline()
    numfiles=0
    while line:
        numfiles=numfiles+1
        username = line.split()[0]
        if username not in setofusers:
            userdict[username]=len(setofusers)
            setofusers.add(username)
        line = f.readline()

    print("No. of users= ",len(setofusers))   
    flag=0
    for i in range(len(results[0])):
        if results[0][i]>0.5:
            pred=i
            probability=results[0][i]
	    flag=1
            break
   
    if flag==1:
    	listOfItems = userdict.items()
    	for item in listOfItems:
           if item[1] == pred:
              prediction=item[0]
              break
    else:
	prediction="NA"
	probability=0

    K.clear_session()
    
    return render_template(
    'submitted_test.html',
    name=name,
    email=email,
    filename=filename,
    prediction=prediction,
    probability=probability)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)


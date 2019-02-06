# aihack
Signature recognition using deep learning model (convolutional neural network layers) implemented using keras deep learning libraries in python

Deploying it is pretty straightforward. In order to deploy the web-app in your system, you need following python libraries: python, tensorflow, keras, opencv, flask and h5py (Anaconda works)

userfilemap.txt contains the database of usernames and signature filenames. TRAIN_FOLDER (genuine) contains the training images. TEST_FOLDER (uploads) contains the test images.

To run the app you have to go to the main folder in cmd and run 'python main.py'. This will launch flask and deploy the main app at http://localhost:5000

1) The training page: http://localhost:5000/train (please read the instructions in page). Files uploaded from this page will go to the 'genuine' folder.

Once you click submit, it will upload your file to 'genuine' folder and add an entry for it in 'userfilemap.txt' and (if Comments='Train') also launch a training. Please note training takes time. Once trained the model will be stored in 'deepwriter.h5' file.

2) The test page: http://localhost:5000/test Files uploaded from this page will go to the 'uploads' folder. On clicking 'Submit' system will predict whose signature it is based on 'deepwriter.h5' model.




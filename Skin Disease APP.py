
#Import necessary libraries
from flask import Flask, render_template, request
 
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
 
#load model
model =load_model("Skin Disease Model.h5", compile=False)
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
 
print('@@ Model loaded')

 
def pred_skin_dieas(skin_dieas):
  test_image = load_img(skin_dieas, target_size = (220, 220)) # load image 
  print("@@ Got Image for prediction")
   
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
  result = model.predict(test_image).round(3) # detect skin diseased
  print('@@ Raw result = ', result)
   
  pred = np.argmax(result) # get the index of max value
 
  if pred == 0:
    return 'Bullous Skin Disease', 'Skin_disease.html' # if index 0 Bullous Disease Photos
  elif pred == 1:
    return 'Seborrheic Keratoses Skin Disease', 'Skin_disease.html' # # if index 1  Seborrheic Keratoses
  else:
    return "Warts Molluscum Skin Disease", 'Skin_disease.html' # if index 3
 
     

# Create flask instance
app = Flask(__name__)
 
# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('Home_Page.html')
     
     
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
         
        file_path = os.path.join('static/Uploaded Images', filename)
        file.save(file_path)
 
        print("@@ Predicting class......")
        pred, output_page = pred_skin_dieas(skin_dieas=file_path)
               
        return render_template(output_page, pred_output = pred, user_image = file_path)
     
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False) 



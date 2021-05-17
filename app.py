import numpy as np
from flask import Flask, request, jsonify,render_template
import pickle
import base64
import cv2
import os
from keras.models import load_model

app = Flask(__name__)


pickle_model = load_model('model.h5')
print("MODEL loaded")

@app.route('/sendImage', methods= ['POST'])
def get_image():
    with open("image (991).JPG") as image_file:
        data = base64.b64encode(image_file.read())
    dict={}
    print(data)
    dict['imgEncoding']= str("Blah Blah")
    return dict
    

@app.route('/api',methods=['POST'])
def predict():
    image = request.files["images"]
    image_name = image.filename
    image.save(os.path.join(os.getcwd(), image_name))
    img = cv2.imread(image_name)
    default_image_size = tuple((256, 256))
    img = cv2.resize(img,default_image_size)   
    #img = img_to_array(img)
    #img = img/1.0
    np_image =  np.array(img, dtype=np.float16) / 255.0
    np_image = np_image.reshape((1, 256, 256, 3))
    print(np_image)
    Ypredict = pickle_model.predict(np_image)
    LABELS = ['Apple___Apple_Scab', 'Apple___Black_Rot', 'Apple___Cedar_Apple_Rust', 'Apple___healthy', 'Arjun___diseased', 'Arjun___healthy', 'Background_without_leaves', 'Basil___healthy', 'Blueberry___healthy', 'Cassava___Bacterial_Blight', 'Cassava___Brown_Streak_Disease', 'Cassava___Green_Mottle', 'Cassava___Healthy', 'Cassava___Mosaic_Disease', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Chinar___diseased', 'Chinar___healthy', 'Corn___Cercospora_leaf_spot, Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Daisy', 'Dandelion', 'Gauva___diseased', 'Gauva___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Jamun___diseased', 'Jamun___healthy', 'Lemon___diseased', 'Lemon___healthy', 'Mango___diseased', 'Mango___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Pomegranate___diseased', 'Pomegranate___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Rose', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Sunflower', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy', 'Tulip']
    #print(Ypredict)
    output = LABELS[Ypredict.argmax()]
    print(output)
    print(Ypredict.argmax())
    ret = {'class':output}
    return jsonify(ret)


@app.route('/test',methods=['GET'])
def hello_world():
    dict = {}
    dict['Query'] = str(request.args['Query'])
    # print(jsonify(dict))
    return jsonify(dict)

@app.route('/')
def hello_html():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
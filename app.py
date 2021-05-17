import numpy as np
from flask import Flask, request, jsonify,render_template
import pickle
import base64
import cv2
import urllib.request
import os
from io import BytesIO
from scipy import misc
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
from tensorflow.keras.models import Sequential, Model
import dill

app = Flask(__name__)





def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
# def make_keras_picklable():

#     def __reduce__(self):
#         model_metadata = saving_utils.model_metadata(self)
#         training_config = model_metadata.get("training_config", None)
#         model = serialize(self)
#         weights = self.get_weights()
#         return (unpack, (model, training_config, weights))

#     cls = Model
#     cls.__reduce__ = __reduce__



class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "unpack":
            module = "unpack"
        return super().find_class(module, name)

with open('MobileNet_200_30ep_50BS_aug_8_1_1.pkl', 'rb') as f:
    # unpickler = MyCustomUnpickler(f)
    pickle_model = dill.load(f)


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
    np_image =  np.array(img, dtype=np.float16) / 225.0
    np_image = np_image.reshape((1, 256, 256, 3))
    print(np_image)
    # Make prediction using model loaded from disk as per the data.
    
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

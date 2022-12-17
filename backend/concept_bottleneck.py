import numpy as np
from flask import Flask, request,render_template, jsonify
from load_model import *
from predict_rerun import *
from PIL import Image
import base64
import io
import json



app = Flask(__name__,template_folder='../templates')

con_model = define_con_model()
mlp_model = define_mlp_model()
concept =dict()
classify = dict()
img_data = None

attributes = []
with open("../CUB_200_2011/attributes.txt", 'r') as f:
    for line in f:
        items = line.strip().split()
        attributes.append(items[1])


@app.route('/')
def index():
    img_data= None
    concept ={}
    classify ={}

    return render_template('BottleNeckUI.html',concept=concept, classify=classify, img_data=img_data)


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    global img_data
    global concept
    global classify

    image = request.files["img"]
    print("image path:", image)
    im = Image.open(image)
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    result, classify = predict_bird(image)

    concept_no = 1
    for k, v in result.items():
        concept[concept_no] = {"attribute":k, "value":v}
        concept_no += 1

    img_data=encoded_img_data.decode('utf-8')

    classify = dict(sorted(classify.items(), key=lambda item: item[1], reverse=True))
    concept = dict(sorted(concept.items(), key=lambda item: item[1]['value'], reverse=True))

    return render_template('BottleNeckUI.html',concept=concept, classify=classify, img_data=img_data)



@app.route('/rerun', methods=['POST'])
def rerun():
    global img_data
    global concept
    global classify
    '''
    For rendering results on HTML GUI
    '''

    new_concepts = request.form.to_dict()

    new_concepts = {int(k):float(v.strip()) for k,v in new_concepts.items()}
    print("old_concepts:", new_concepts)
## sort the data based on index value in ascending and then create json
    new_concepts = dict(sorted(new_concepts.items()))
    print("new_concepts",new_concepts)

    new_attr_concept = {}
    for k, v in new_concepts.items():
        new_attr_concept[attributes[k-1]]= float(v)

    result, classify = rerun_class(new_attr_concept)

    concept_no = 1
    for k, v in result.items():
        concept[concept_no] = {"attribute":k, "value":v}
        concept_no += 1
    
    classify = dict(sorted(classify.items(), key=lambda item: item[1], reverse=True))
    concept = dict(sorted(concept.items(), key=lambda item: item[1]['value'], reverse=True))
    
    return render_template('BottleNeckUI.html',concept=concept, classify=classify, img_data=img_data)


if __name__ == "__main__":

    app.run(debug=True)
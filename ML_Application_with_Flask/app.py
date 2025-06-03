from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
        For rendering results on HTML GUI
    '''
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = model.predict(final_features)

    return render_template('index.html',prediction_text = 'Predicted class:{}'.format(prediction))




if __name__ == "__main___":
    app.run(debug=True)
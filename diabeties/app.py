from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# __name__ tells Flask where your current Python script is located, so Flask knows how to find important folders like:

# templates/ (for your HTML files)

# static/ (for CSS, JavaScript, images, etc.)

# app = Flask(__name__, template_folder='htmlfile',     # Custom HTML folder
#             static_folder='css_static' )     # Custom folder for CSS/JS/images


model = pickle.load(open('dia.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = model.predict(final_features)

    result = " You have Diabetes:(" if prediction[0] == 1 else " No diabeties:)"

    return render_template('index.html',prediction_text = f'Predicted class:{result}')
    


if __name__ == "__main__":
    app.run(debug=True)
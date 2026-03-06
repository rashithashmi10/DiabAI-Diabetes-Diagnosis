from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)
app.config['SECRET_KEY'] = '5159'

scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('rf_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = -1
    if request.method == 'POST':
        # Extract form data
        pregs = int(request.form.get('pregs'))
        gluc = int(request.form.get('gluc'))
        bp = int(request.form.get('bp'))
        skin = int(request.form.get('skin'))
        insulin = float(request.form.get('insulin'))
        bmi = float(request.form.get('bmi'))
        func = float(request.form.get('func'))
        age = int(request.form.get('age'))

        # Prepare input features
        input_features = [[pregs, gluc, bp, skin, insulin, bmi, func, age]]
        prediction = model.predict(scaler.transform(input_features))[0]

        # Check if request is AJAX
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'prediction': int(prediction)})

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__, template_folder='template')

# Load the models
model1 = pickle.load(open('CT.pkl', 'rb'))
model2 = pickle.load(open('PS.pkl', 'rb'))#PS
model3 = pickle.load(open('ITSdry.pkl', 'rb'))#ITS DRY
model4 = pickle.load(open('ITSwet.pkl', 'rb'))#ITS wet
AV = pickle.load(open('AV.pkl', 'rb'))
Gmb = pickle.load(open('Gmb.pkl', 'rb'))
VMA = pickle.load(open('VMA.pkl', 'rb'))
VFB = pickle.load(open('VFB.pkl', 'rb'))
OBC = pickle.load(open('OBC.pkl', 'rb'))

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/team')
def team():
    return render_template('team.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_model = request.form['model_select']
        if selected_model == 'model1':
            return redirect(url_for('rutting', model='model1'))
        elif selected_model == 'model2':
            return redirect(url_for('rutting', model='model2'))
        elif selected_model == 'model3' and 'model4':
            return redirect(url_for('moisture', model='model3_model4'))
        elif selected_model == 'AV' and 'Gnb' and 'VMA'and 'VFB' and 'OBC':
            return redirect(url_for('volumetrics', model='volumetrics'))
        
    return render_template('base.html')

@app.route('/rutting/<model>', methods=['GET', 'POST'])
def rutting(model):
    pred1, pred2 = None, None
    
    if request.method == 'POST':
        aggregate = request.form['Aggregate']
        source = request.form['source']
        viscosity = float(request.form['viscosity'])  # Assuming viscosity is a float
        dag = request.form['DAG']
        air_voids = request.form['air_voids']
        
        # Convert input data to a format that the model expects
        input_data = np.array([[aggregate, source, viscosity, dag, air_voids]])
        
        # Make prediction using the selected model
        if model == 'model1':
            pred1 = model1.predict(input_data)
            pred1 = [f"{float(pred1)-2:.2f}", f"{float(pred1)+2:.2f}"]
        elif model == 'model2':
            pred2 = model2.predict(input_data)
            pred2 = [f"{float(pred2)-0.2:.2f}", f"{float(pred2)+0.2:.2f}"]
        
    return render_template('rutting.html', model=model, pred1=pred1, pred2=pred2)

@app.route('/moisture/<model>', methods=['GET', 'POST'])
def moisture(model):
    pred3, pred4, tsr = None, None, None
    
    if request.method == 'POST':
        aggregate = request.form['Aggregate']
        source = request.form['source']
        viscosity = float(request.form['viscosity'])  # Assuming viscosity is a float
        dag = request.form['DAG']
        air_voids = request.form['air_voids']
        
        # Convert input data to a format that the model expects
        input_data = np.array([[aggregate, source, viscosity, dag, air_voids]])
        
        # Make prediction using the selected models
        if model == 'model3_model4':
            pred3 = np.round(model3.predict(input_data), 2)
            pred4 = np.round(model4.predict(input_data), 2)
            tsr = np.round(pred4 * 100 / pred3, 2) if pred3.all() != 0 else "Invalid prediction: Division by zero"

    return render_template('moisture.html', model=model, pred3=pred3, pred4=pred4, tsr=tsr)

@app.route('/volumetrics/<model>', methods=['GET', 'POST'])
def volumetrics(model):
    pred6, pred7, pred8, pred9, pred0 = None, None, None, None, None  # Initialize predictions

    if request.method == 'POST':
        # Retrieve form data
        aggregate = request.form['Aggregate']
        source = request.form['source']
        
        try:
            viscosity = float(request.form['viscosity'])  # Convert viscosity to float
        except ValueError:
            return render_template('volumetrics.html', model=model, error="Viscosity must be a number.")
        
        dag = request.form['DAG']
        compaction = request.form['compaction']
        
        # Convert input data to a format that the model expects
        input_data = np.array([[aggregate, source, viscosity, dag, compaction]])
        input_data1 = np.array([[aggregate, source, viscosity, dag]])

        # Make prediction using the selected models
        if model == 'volumetrics':
            pred6 = np.round(AV.predict(input_data), 2)
            pred7 = np.round(Gmb.predict(input_data), 2)
            pred8 = np.round(VMA.predict(input_data), 2)
            pred9 = np.round(VFB.predict(input_data), 2)
            pred0 = np.round(OBC.predict(input_data1), 2)
            
    return render_template('volumetrics.html', model=model, pred6=pred6, pred7=pred7, pred8=pred8, pred9=pred9, pred0=pred0)

if __name__ == '__main__':
    app.run(debug=True)

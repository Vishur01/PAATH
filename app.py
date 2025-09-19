from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__, template_folder='template')

# Load the models
CT_pg = pickle.load(open('CT.pkl', 'rb'))
CT_sp= pickle.load(open('CT_SP.pkl', 'rb'))
PS_pg = pickle.load(open('PS.pkl', 'rb'))#PS
PS_sp = pickle.load(open('PS_sp.pkl', 'rb'))#PS
ITSdry = pickle.load(open('ITSdry.pkl', 'rb'))#ITS DRY
ITSwet = pickle.load(open('ITSwet.pkl', 'rb'))#ITS wet
ITSdry_sp = pickle.load(open('ITSdry_sp.pkl', 'rb'))#ITS DRY
ITSwet_sp = pickle.load(open('ITSwet_sp.pkl', 'rb'))#ITS wet
AV = pickle.load(open('AV.pkl', 'rb'))
Gmb = pickle.load(open('Gmb.pkl', 'rb'))
VMA = pickle.load(open('VMA.pkl', 'rb'))
VFB = pickle.load(open('VFB.pkl', 'rb'))
OBC = pickle.load(open('OBC.pkl', 'rb'))
AV_sp = pickle.load(open('AV_sp.pkl', 'rb'))
Gmb_sp = pickle.load(open('Gmb_sp.pkl', 'rb'))
VMA_sp = pickle.load(open('VMA_sp.pkl', 'rb'))
VFB_sp = pickle.load(open('VFB_sp.pkl', 'rb'))
OBC_sp = pickle.load(open('OBC_sp.pkl', 'rb'))

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
        if selected_model == 'model1'and 'model11' :
            return redirect(url_for('rutting', model='model1'))
        elif selected_model == 'model2'and 'model22':
            return redirect(url_for('strain', model='model2'))
        elif selected_model == 'model3' and 'model4':
            return redirect(url_for('moisture', model='model3_model4'))
        elif selected_model == 'AV' and 'Gnb' and 'VMA'and 'VFB' and 'OBC':
            return redirect(url_for('volumetrics', model='volumetrics'))
        
    return render_template('base.html')


@app.route('/rutting/<model>', methods=['GET', 'POST'])
def rutting(model):
    pred1, pred11 = None, None
    
    if request.method == 'POST':
        aggregate = request.form['Aggregate']
        source = request.form['source']
        binder_input_type = request.form['binder_input_type']
        dag = request.form['DAG']
        air_voids = request.form['air_voids']

        if binder_input_type == 'pg':
            binder_value = float(request.form['viscosity'])
            # Prepare feature vector and predict using CT_pg
            input_vector = [[aggregate, source, binder_value, dag, air_voids]]
            pred1 = CT_pg.predict(input_vector)
            pred1 = [f"{float(pred1)-2:.2f}", f"{float(pred1)+2:.2f}"]
            # pass pred1 to template

        elif binder_input_type == 'softening':
            binder_value = float(request.form['softening_point'])
            # Prepare feature vector and predict using CT_sp
            input_vector = [[aggregate, source, binder_value, dag, air_voids]]
            pred11 = CT_sp.predict(input_vector)
            pred11 = [f"{float(pred11)-2:.2f}", f"{float(pred11)+2:.2f}"]
            # pass pred11 to template
        
    return render_template('rutting.html', model=model, pred1=pred1, pred11=pred11)

@app.route('/strain/<model>', methods=['GET', 'POST'])
def strain(model):
    pred2, pred22 = None, None
    
    if request.method == 'POST':
        aggregate = request.form['Aggregate']
        source = request.form['source']
        binder_input_type = request.form['binder_input_type']
        dag = request.form['DAG']
        air_voids = request.form['air_voids']

        if binder_input_type == 'pg':
            binder_value = float(request.form['viscosity'])
            # Prepare feature vector and predict using CT_pg
            input_vector = [[aggregate, source, binder_value, dag, air_voids]]
            pred2 = PS_pg.predict(input_vector)
            pred2 = [f"{float(pred2)-2:.2f}", f"{float(pred2)+2:.2f}"]
            # pass pred1 to template

        elif binder_input_type == 'softening':
            binder_value = float(request.form['softening_point'])
            # Prepare feature vector and predict using CT_sp
            input_vector = [[aggregate, source, binder_value, dag, air_voids]]
            pred22 = PS_sp.predict(input_vector)
            pred22 = [f"{float(pred22)-2:.2f}", f"{float(pred22)+2:.2f}"]
            # pass pred11 to template
        
    return render_template('strain.html', model=model, pred2=pred2, pred22=pred22)


@app.route('/moisture/<model>', methods=['GET', 'POST'])
def moisture(model):
    pred3, pred4, tsr1, pred33, pred44, tsr2 = None, None, None, None, None, None
    
    if request.method == 'POST':
        aggregate = request.form['Aggregate']
        source = request.form['source']
        binder_input_type = request.form['binder_input_type']
        dag = request.form['DAG']
        air_voids = request.form['air_voids']
        
        if binder_input_type == 'pg':
            binder_value = float(request.form['viscosity'])
            
            input_vector = [[aggregate, source, binder_value, dag, air_voids]]
            pred3 = np.round(ITSdry.predict(input_vector), 2)
            pred4 = np.round(ITSwet.predict(input_vector), 2)
            tsr1 = np.round(pred4 * 100 / pred3, 2) if pred3.all() != 0 else "Invalid prediction: Division by zero"

      

        elif binder_input_type == 'softening':
            binder_value = float(request.form['softening_point'])
        
            input_vector = [[aggregate, source, binder_value, dag, air_voids]]
            pred33 = np.round(ITSdry_sp.predict(input_vector), 2)
            pred44 = np.round(ITSwet_sp.predict(input_vector), 2)
            tsr2 = np.round(pred44 * 100 / pred33, 2) if pred33.all() != 0 else "Invalid prediction: Division by zero"
          
        

    return render_template('moisture.html', model=model, pred3=pred3, pred4=pred4, tsr1=tsr1, pred33=pred33, pred44=pred44, tsr2=tsr2)

@app.route('/volumetrics/<model>', methods=['GET', 'POST'])
def volumetrics(model):
    pred6, pred7, pred8, pred9, pred0, pred66, pred77, pred88, pred99, pred00 = None, None, None, None, None, None, None, None, None, None  

    if request.method == 'POST':
        aggregate = request.form['Aggregate']
        source = request.form['source']
        binder_input_type = request.form['binder_input_type']
        dag = request.form['DAG']
        compaction = request.form['compaction']
        
       
        if binder_input_type == 'pg':
            binder_value = float(request.form['viscosity'])
            
            input_vector = [[aggregate, source, binder_value, dag, compaction]]
            input_vector1 = [[aggregate, source, binder_value, dag]]
            pred6 = np.round(AV.predict(input_vector), 2)
            pred7 = np.round(Gmb.predict(input_vector), 2)
            pred8 = np.round(VMA.predict(input_vector), 2)
            pred9 = np.round(VFB.predict(input_vector), 2)
            pred0 = np.round(OBC.predict(input_vector1), 2)

      

        elif binder_input_type == 'softening':
            binder_value = float(request.form['softening_point'])
        
            input_vector = [[aggregate, source, binder_value, dag, compaction]]
            input_vector1 = [[aggregate, source, binder_value, dag]]
            pred66 = np.round(AV_sp.predict(input_vector), 2)
            pred77 = np.round(Gmb_sp.predict(input_vector), 2)
            pred88 = np.round(VMA_sp.predict(input_vector), 2)
            pred99 = np.round(VFB_sp.predict(input_vector), 2)
            pred00 = np.round(OBC_sp.predict(input_vector1), 2)

        
    return render_template('volumetrics.html', model=model, pred6=pred6, pred7=pred7, pred8=pred8, pred9=pred9, pred0=pred0, pred66=pred66, pred77=pred77, pred88=pred88, pred99=pred99, pred00=pred00)
6
if __name__ == '__main__':
    app.run(debug=True)

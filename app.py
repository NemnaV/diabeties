# импортируем Flask
from flask import Flask, request, render_template
# создаем приложение
app = Flask(__name__)

import pandas as pd
import pickle

# для отслеживания URL-адреса главной страницы:
@app.route('/', methods=['POST', 'GET'])
@app.route('/index',  methods=['POST', 'GET'])
def index():
    return render_template("index.html")

@app.route('/form',  methods=['POST', 'GET'])
def form():
    def pred_new(df_test, sm):
        model = pickle.loads(sm)
        x_raw = df_test.iloc[:, :].values
        new_class = model.predict(x_raw)
        return new_class
    

    if request.method == "POST":
        if request.form['submit_button'] == 'get_results':

            Pregnancies = float(request.form.get('Pregnancies'))
            Glucose = float(request.form.get('Glucose'))
            BloodPressure = float(request.form.get('BloodPressure'))
            SkinThickness = float(request.form.get('SkinThickness'))
            Insulin = float(request.form.get('Insulin'))
            Height = float(request.form.get('Height'))
            Weight = float(request.form.get('Weight'))
            BMI = Weight / ((Height/100)**2)
            DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
            Age = float(request.form.get('Age'))

            data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
            df = pd.DataFrame(data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

            model_name = request.form.get('method')

            lr_model_5_b = pickle.load(open('/Users/valeria/диплом/сайт/lr_model_5_b.pkl', 'rb'))
            lr_model_dpf_b = pickle.load(open('/Users/valeria/диплом/сайт/lr_model_dpf_b.pkl', 'rb'))
            lr_models = [lr_model_5_b, lr_model_dpf_b]

            nb_rfe_model = pickle.load(open('/Users/valeria/диплом/сайт/nb_rfe_model.pkl', 'rb'))
            nb_5_model = pickle.load(open('/Users/valeria/диплом/сайт/nb_5_model.pkl', 'rb'))
            nb_models = [nb_rfe_model, nb_5_model]

            rf_best_model = pickle.load(open('/Users/valeria/диплом/сайт/rf_best_model.pkl', 'rb'))
            rf_dgf_model = pickle.load(open('/Users/valeria/диплом/сайт/rf_dgf_model.pkl', 'rb'))
            rf_models = [rf_best_model, rf_dgf_model]

            if model_name=="lr":
                model = lr_models
                ac_model = [0.7551, 0.7305]
            elif model_name=="nb":
                model = nb_models
                ac_model = [0.7246, 0.7293]
            elif model_name=="rf":
                model = rf_models
                ac_model = [0.7111, 0.7198]
            
            if DiabetesPedigreeFunction == 0 or SkinThickness == 0 or Insulin == 0:
                df = df.drop(['DiabetesPedigreeFunction', 'SkinThickness', 'Insulin'], axis=1)
                result = pred_new(df, model[1])
                ac = ac_model[1]
            elif model_name == "lr" or model_name == "nb":
                df = df.drop(['Age', 'SkinThickness', 'Insulin'], axis=1)
                result = pred_new(df, model[0])
                ac = ac_model[0]
            elif model_name == "rf":
                result = pred_new(df, model[0])
                ac = ac_model[0]
                
            ac = round(ac, 2)*100

    return render_template("index.html", result=result, ac=ac)

# запуск Flask-приложения:
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange
import numpy as np 
from tensorflow.keras.models import load_model
import joblib
from keras import backend as K
import tensorflow as tf
import pickle
import os

port = int(os.environ.get("PORT", 5000))
def return_prediction(model,scaler,sample_json):
    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']
    #print (p_wid )

    flower = [[s_len,s_wid,p_len,p_wid]]
    flower = scaler.transform(flower)
    #print (flower)

    classes = np.array(['setosa', 'versicolor', 'virginica'])
    class_ind = model.predict_classes(flower)
    

    return classes[class_ind][0]


app = Flask(__name__)
# Configure a secret SECRET_KEY
app.config['SECRET_KEY'] = 'someRandomKey'
# Loading the model and scaler
flower_model = load_model('final_iris_model.h5')
#K.clear_session()
#flower_model = pickle.load(open('model.pkl','rb'))

flower_scaler = joblib.load('iris_scaler.pkl')
#graph = tf.get_default_graph()


# Now create a WTForm Class
class FlowerForm(FlaskForm):
    sep_len = TextField('Sepal Length')
    sep_wid = TextField('Sepal Width')
    pet_len = TextField('Petal Length')
    pet_wid = TextField('Petal Width')
    submit = SubmitField('Analyze')
 
@app.route('/', methods=['GET', 'POST'])
def index():
    # Create instance of the form.
    form = FlowerForm()
    
    # If the form is valid on submission
    if form.validate_on_submit():
        # Grab the data from the input on the form.
        session['sep_len'] = form.sep_len.data
        session['sep_wid'] = form.sep_wid.data
        session['pet_len'] = form.pet_len.data
        session['pet_wid'] = form.pet_wid.data
        return redirect(url_for('prediction'))
    return render_template('home.html', form=form)

@app.route('/prediction')
def prediction():
    #Defining content dictionary
    content = {}
    content['sepal_length'] = float(session['sep_len'])
    content['sepal_width'] = float(session['sep_wid'])
    content['petal_length'] = float(session['pet_len'])
    content['petal_width'] = float(session['pet_wid'])
 
    results = return_prediction(model=flower_model,scaler=flower_scaler,sample_json=content)
    return render_template('prediction.html',results=results)
##
##if __name__ == '__main__':
## app.run(debug=True)
#server = app.server
if __name__ == "__main__": 
    app.run(debug=False,
                   host="0.0.0.0",
                   port=port) # at the end of app.py file


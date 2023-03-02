from flask import *

app = Flask(__name__)
nithin = 'Naveen'
@app.route('/')
def home():
    title='Welcome'
    return render_template('home.html')

@app.route('/process_form',methods=['GET','POST'])
def process_form():
    if request.method == 'POST':
        nithin = request.form['name']
        print(nithin)
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)

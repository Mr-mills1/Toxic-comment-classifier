# import the libraries
from flask import Flask, render_template, request, redirect, url_for
from joblib import load
import pandas as pd
pd.set_option('display.max_colwidth', 1000) # set column width

# load the saved or dumped pipeline
pipeline = load("toxic_comment_classification.joblib")

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    pred=None
    c=[]
    col=['severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
    if request.method == 'POST':
        user_input = request.form['comment']
        print(user_input)
        cat=pipeline.predict([user_input])
        print(cat)
        cat = pd.DataFrame(cat,columns=col)
        for i in range(len(cat)):
            if cat.columns[(cat == 1).iloc[i]].notna().all():
                c=(cat.columns[(cat == 1).iloc[i]].values)
                print(c)
    return render_template('index.html',k=c)

if __name__ == '__main__':
    app.run(debug=True)
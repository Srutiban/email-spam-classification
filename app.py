from flask import Flask,render_template,request
import pickle

cv = pickle.load(open("model/vectorizer.pkl","rb"))
clf = pickle.load(open("model/model.pkl","rb"))

app = Flask(__name__)

@app.route('/')
def index():
    #vectorize the input
    #result = cv.transform([sample]).toarray()
    #predict
    #pred = clf.predict(result)
    #print(pred)
    return render_template("index.html")

@app.route('/predict',methods=['post'])
def predict():
    userInput = request.form.get('email')
    result = cv.transform([userInput]).toarray()
    # predict
    pred = clf.predict(result)
    pred = int(pred[0])
    if pred == 0:
        pred = -1
    return render_template("index.html",label=pred)


if __name__ == "__main__":
    app.run(debug=True)
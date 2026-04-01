from flask import Flask

app = Flask(__name__)

@app.route('/predict',methods=["GET"] )
def predict():
    return "Wow, there are a get route with Flask 😯😯😯 (yes, is that easy) "

if __name__ == "__main__":
    app.run(debug=True)
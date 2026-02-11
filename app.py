from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("emotion_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    emotion = ""
    if request.method == "POST":
        text = request.form["text"]
        emotion = model.predict([text])[0]
    return render_template("index.html", emotion=emotion)

if __name__ == "__main__":
    app.run()

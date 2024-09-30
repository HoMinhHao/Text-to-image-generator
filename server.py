from flask import Flask, request, render_template
from text2img import *

app=Flask(__name__)

SAVE_IMAGE_PATH='static/output.jpg'

pipeline=generate()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('index.html')
    else:
        user_input=request.form['prompt']
        print("Start generate....")
        img=text2img(user_input,pipeline)
        print("Finist generate")
        img.save(SAVE_IMAGE_PATH)
        return render_template('index.html', image_url=SAVE_IMAGE_PATH)
    
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
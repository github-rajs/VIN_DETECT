import flask
from flask import Flask, render_template
import os
import raj
import json

app = Flask(__name__, static_folder="build/static",
 template_folder="build")

@app.route("/")
def root():
    return "Welcome page"
	
@app.route("/API_EXAMPLE/<path:path>")
def claimpredict_class(path):
    print("the path  is .......",path)
    image = path
    print(image)
    liv3,pred3=raj.fake_image_detector_two(image);
    if liv3 == 'fake':
        result_jason = {
        'fake_detect' : [
            {
                'result' : liv3,
                'score' : pred3
            },
        ],
        }

    else:
        txts,score=raj.vin_main_fn(image);

        result_jason = {
            'fake_detect' : [
                {
                    'result' : liv3,
                    'score' : pred3
                },

            ],
                'vin_rec' : [
                {
                    'result' : txts,
                    'score' : score
                }

            ]
        }

        # model_name = "ocr_model"
        # ocr_model_output = model_name.predict_proba(img_name)

        #ocr_model_output="MNBFR457889"

    return json.dumps(str(result_jason))

def before_request():
    app.jinja_env.cache = {}    

if __name__ == '__main__':
    int()
    port = os.environ.get('PORT', 5002)
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False, host='127.0.0.1', port=port)
    print('flask server running!')

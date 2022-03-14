import flask
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os
import logging
from datetime import datetime
from PIL import Image
import raj
import json

app = Flask(__name__,template_folder="templates")
file_handler = logging.FileHandler('server.log')
logging.basicConfig(filename='record.log', level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

@app.route("/")
def root():
    return "Welcome page"
	
@app.route('/', methods = ['POST','GET'])
def api_root():
    if request.method == 'POST' and request.files['image']:
      app.logger.info("POST request recieved")
      form_data = request.form
      #result=request.form.to_dict(flat=False)
      img = request.files['image']
      app.logger.info("Recieved image")
      img.save('temp_image/rcvd.jpg')
      app.logger.info("saved image")
      img_name = secure_filename(img.filename)
      imga = Image.open(img.stream)
      app.logger.info("read image successfull")
      app.logger.info("sending image to fake image detection model")
      liv3,pred3,im_remarks=raj.fake_image_detector_two('temp_image/rcvd.jpg');
      if liv3 == 'fake':
        app.logger.info("fake image detected.")
        result_jason = {'result':liv3,'image_accuracy':pred3,'Image Rearks':im_remarks}
      else:
        app.logger.info("real image detected.")
        txts,score=raj.vin_main_fn('temp_image/rcvd.jpg');
        vin='null' if not txts else txts[0].replace(" ", "")
        dig='null' if not score else str(float(score[0]).__round__(2))
        first_five_letters=vin[0:5]  
        if  vin == 'null' or dig =='null':
          remarks="VIN recognition failed due to bad text quality:Please enter the VIN Manulally" 
        elif float(dig) < 0.50 or int(vin.__len__()) < 12:
          remarks="Bad text clarity or one or more texts are damaged :Enter VIN number manually "
        elif int(vin.__len__()) == 17 and float(dig) > 0.80:
          remarks="All VIN characters recognised. Please re-verify the VIN"
        elif int(vin.__len__()) == 17 and float(dig) > 0.80 and first_five_letters== "MBLHA" or first_five_letters == "MBLJA":
          remarks="VIN detection complete.first 5 letters:{first_five_letters}"
        elif int(vin.__len__()) == 17 and float(dig) > 0.80 and str(first_five_letters)!= "MBLJA" or str(first_five_letters) != "MBLHA":  
          remarks="Some characters are recognised incorrectly:Please correct the VIN manually.errors:{first_five_letters}"
        elif float(dig) > 0.80 and int(vin.__len__()) < 17:
          remarks="One or more VIN charactertes not recognised:Enter VIN Manually"
        app.logger.info("VIN recognition complete!")
        result_jason = {'result':liv3,'image_accuracy':pred3,'data':vin,'data_accuracy':dig,'Image Remarks':im_remarks,'VIN Remarks':remarks}
        app.logger.info("sending result to client")
      return render_template("output.html",result = result_jason)
    else:
       app.logger.info("No image found")
       return "Where is the image?"

    return json.dumps(str(result_jason))

def before_request():
    app.jinja_env.cache = {}    

if __name__ == '__main__':
    int()
    port = os.environ.get('PORT', 5002)
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False, host='127.0.0.1', port=port)
    app.logger.critical("Flask server is down")

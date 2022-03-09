from paddleocr import PaddleOCR,draw_ocr
import PIL
from PIL import ImageFont
from PIL import Image
import PIL.Image
from imutils import paths
import cv2
from matplotlib import pyplot as plt
import json
import os
import shutil
import pandas as pd
from pylab import imshow
import numpy as np
import torch
import seaborn as sns
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import load_rgb
from datasouls_antispoof.pre_trained_models import create_model
from datasouls_antispoof.class_mapping import class_mapping
from imutils.video import VideoStream
import tensorflow as tf
import argparse
import imutils
import pickle
import time 
import tensorflow_hub as hub
#@torch.no_grad()


### Paddle OCR Model ###
def vin_ocr_model(img_path):
    ocr = PaddleOCR(det_model_dir='paddleOCR_V2_EN_GEN_SRV_MOB_T/ch_PP-OCRv2_det_distill_train', rec_model_dir='paddleOCR_V2_EN_GEN_SRV_MOB_T/ret_model/ch_PP-OCRv2_rec_train',cls_model_dir='paddleOCR_V2_EN_GEN_SRV_MOB_T/cls_model/ch_ppocr_mobile_v2.0_cls_train',use_angle_cls=True) ;
    result = ocr.ocr(img_path,cls=True);
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    return txts,scores

### Main Code ###
def vin_main_fn(img):
    isdir = os.path.isdir('runs/detect')
    if isdir == True:
        shutil.rmtree(r'runs/detect')
    else:
        pass

    read_img = cv2.imread(img)
    gray_image=cv2.cvtColor(read_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('temp/gray_image.jpg', gray_image)
    
    #Detect VIN Region
    #exec(open('hero.py').read())
    #os.system('python yolov5/detect.py --weights yolov5/yolov5/weights/large/best.pt --img 416 --conf 0.6 --source temp/gray_image.jpg --project runs/detect/ --name exp --save-crop')
    torch.hub.set_dir('L:/TEMP')
    #model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/yolov5/weights/best.pt').to("cpu")  # default
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='cpu_model.pt').to("cpu")
    imgs = ['temp/gray_image.jpg'] 
    results = model(imgs)
    results.xyxy[0]  
    results.pandas().xyxy[0]  
    crops = results.crop(save=True)
#
    ##Check if directry present(True if object detection is correct)
    detect_folder='runs/detect/exp/crops/Letters/'
    isdir = os.path.isdir(detect_folder)
    if isdir == False:
        txts,score=vin_ocr_model(img)   
    else:
        print('Dir exist')
        file = os.listdir(detect_folder)
        file=file[0]
        path = (detect_folder+""+file)
        src = cv2.imread(path)
        image = cv2.rotate(src, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite("runs/temp/"+file, image)    
        #Convert to inverted imge
        gray_image_file='runs/temp/'+ file
        image_read=cv2.imread(gray_image_file)
        inverted_image = cv2.bitwise_not(image_read)
        cv2.imwrite("runs/temp/inverted.jpg", inverted_image)
        #Pass images to OCR
        txts,score=vin_ocr_model("runs/temp/inverted.jpg")
        
    torch.cuda.empty_cache()
    
    return txts,score


def fake_image_detector_one(img_in):
    ##Livesness detection
    ##Datasouls
    ###   
    #model = create_model("tf_efficientnet_b3_ns")
    model1 = create_model("tf_efficientnet_b3_ns")
    model2 = create_model("swsl_resnext50_32x4d")
    model1.eval();
    model2.eval();
    image_replay = load_rgb(img)
    #imshow(image_replay)
    ###
    transform = albu.Compose([albu.PadIfNeeded(min_height=400, min_width=400),
                              albu.CenterCrop(height=400, width=400), 
                              albu.Normalize(p=1), 
                              albu.pytorch.ToTensorV2(p=1)], p=1)
    
    with torch.no_grad():
        prediction1 = model1(torch.unsqueeze(transform(image=image_replay)['image'], 0)).numpy()[0]
        prediction2 = model2(torch.unsqueeze(transform(image=image_replay)['image'], 0)).numpy()[0]

    df1 = pd.DataFrame({"prediction": prediction1, "class_name": class_mapping.keys()})
    df2 = pd.DataFrame({"prediction": prediction2, "class_name": class_mapping.keys()})
    #sns.barplot(data=df1, x="prediction", y="class_name") 
    #sns.barplot(data=df2, x="prediction", y="class_name") 
    
    res1_df=df1.loc[df1['prediction'].idxmax()]
    res2_df=df2.loc[df2['prediction'].idxmax()]
    redf1,redf2,pred1,pred2=res1_df.class_name,res2_df.class_name,res1_df.prediction,res2_df.prediction
    ###
    return redf1,redf2,pred1,pred2
    
    
     
### ivesness detection models ###
    ##Neural Network(Binary classification)

def fake_image_detector_two(img_in):
    liveness_model = tf.keras.models.load_model('LIVENESS_DETECT/face_liveness_detection/model',custom_objects={'KerasLayer':hub.KerasLayer})
    le = pickle.loads(open('LIVENESS_DETECT/face_liveness_detection/label_encoder','rb').read())
    
    ##Read image and resize --
    frame = cv2.imread(img_in)
    face = cv2.resize(frame, (224,224))
    
    face = face.astype('float') / 255.0 
    face = tf.keras.preprocessing.image.img_to_array(face)
    # tf model require batch of data to feed in
    # so if we need only one image at a time, we have to add one more dimension
    # in this case it's the same with [face]
    face = np.expand_dims(face, axis=0)
    
    preds = liveness_model.predict(face)[0]
    j = np.argmax(preds)
    label = le.classes_[j]
    result2= 'fake' if label ==1 else 'real'
    res3df=pd.DataFrame(preds,columns=['dd'])
    tmp_df=res3df.loc[res3df['dd'].idxmax()]
    pred3=tmp_df.dd
    return result2,pred3

########################################
os.environ["CUDA_VISIBLE_DEVICES"]=""



   
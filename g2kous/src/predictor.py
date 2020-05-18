import glob
import os
import cv2
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from model import yolo_eval, yolo_body, tiny_yolo_body
from utils import letterbox_image
from timeit import default_timer as timer

class ScoringService(object):
    IDvalue = 0 # クラス変数を宣言 = 0

    @classmethod
    def get_model(cls, model_path='../model'):
        modelpath = os.path.join(model_path, 'tinyYOLOv3_cl10_val_loss21.h5')

        class_names = cls._get_class()
        anchors = cls._get_anchors()

        # Load model, or construct model and load weights.
        num_anchors = len(anchors)
        num_classes = len(class_names)
        is_tiny_version = num_anchors == 6  # default setting
        cls.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                         if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
        cls.yolo_model.load_weights(modelpath)  # make sure model, anchors and classes match
        return True


    @classmethod
    def predict(cls, input):
        predictions = []
        cap = cv2.VideoCapture(input)
        fname = os.path.basename(input)

        Nm_fr = 0

        while True:
            Nm_fr = Nm_fr + 1

            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(frame)
            print(frame.shape)
            print("PROCESSING FRAME = ", Nm_fr)
            prediction = cls.detect_image(image)

            predictions.append(prediction)

        cap.release()
        return {fname: predictions}

    @classmethod
    def _get_class(cls, model_path='../src'):
        classes_path = os.path.join(model_path, 'voc_10classes.txt')
      
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    @classmethod
    def _get_anchors(cls, model_path='../src'):
        anchors_path = os.path.join(model_path, '2020_yolo_cl10_anchors.txt')
      
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    @classmethod
    def compute_output(cls, image_data, image_shape):
        input_image_shape = tf.constant(image_shape)

        class_names = cls._get_class()
        anchors = cls._get_anchors()
        iou = 0.5    #Adjust param
        score = 0.3  #Adjust param
      
        boxes, scores, classes = yolo_eval(cls.yolo_model(image_data), anchors,
                                             len(class_names), input_image_shape,
                                             score_threshold=score, iou_threshold=iou)
        return boxes, scores, classes
    
    @classmethod
    def detect_image(cls, image):
        start = timer()
      
        model_image_size = (416, 416)
        class_names = cls._get_class()

        new_image_size = (image.width - (image.width % 32),
                                image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
      
        image_shape = [image.size[1], image.size[0]]

        out_boxes, out_scores, out_classes = cls.compute_output(image_data, image_shape)

        Car_result_ALL = []
        Pedestrian_result_ALL = []
        all_result = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            
            label = '{} {:.2f}'.format(predicted_class, score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            #JSON 形式の時はint32()未対応のため -> int()に変換する
            top = int(top)
            left = int(left)
            bottom = int(bottom)
            right = int(right)
         
            #1 予測結果より次のFrameの物体位置を予測
            #nxt_result_txt = ' {},{},{},{},{}'.format(left, top, right, bottom, c)
            
            #2 検出したbox_sizeを計算する　設定した閾値1024pix**2
            sq_bdbox = (bottom - top)*(right - left) 

            if sq_bdbox >= 1024:#矩形サイズの閾値
                #検出しない時の初期値
                Car_result = {'id': int(0), 'box2d': [int(0),int(0),int(image.height),int(image.width)]}
                Pedestrian_result = {'id': int(0), 'box2d': [int(0),int(0),int(image.height),int(image.width)]}

                if predicted_class == 'Car':
                    cls.IDvalue = cls.IDvalue + 1
                    #車を検出した時
                    Car_result = {'id': int(cls.IDvalue), 'box2d': [left,top,right,bottom]}#予測結果

                    #検出したオブジェクトを格納 検出しない場合は初期値０が格納される
                    Car_result_ALL.append(Car_result)#車
                    Pedestrian_result_ALL.append(Pedestrian_result)#歩行者
                  
                elif predicted_class == 'Pedestrian':
                    cls.IDvalue = cls.IDvalue + 1
                    #歩行者を検出した時
                    Pedestrian_result = {'id': int(cls.IDvalue), 'box2d': [left,top,right,bottom]}#予測結果
              
                    #検出したオブジェクトを格納 検出しない場合は初期値０が格納される
                    Car_result_ALL.append(Car_result)#車
                    Pedestrian_result_ALL.append(Pedestrian_result)#歩行者
        
        all_result = {'Car': Car_result_ALL, 'Pedestrian': Pedestrian_result_ALL}
        end = timer()
        print("1フレームの処理時間 = ", end - start)
        return all_result
    

import glob
import cv2
import pickle

import colorsys

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from model import yolo_eval, yolo_body, tiny_yolo_body
from utils import letterbox_image
from timeit import default_timer as timer

import os

class ScoringService(object):
    IDvalue = 0 # クラス変数を宣言 = 0
    Mem_IDvalue = 0 # クラス変数を宣言 = 0

    @classmethod
    def get_model(cls, model_path='../model'):
        cls.IDvalue = 0 # Reset Object ID
        cls.Mem_IDvalue = 0 # Reset Memory Object ID

        modelpath = os.path.join(model_path, 'YOLOv3_cl10_val_loss67.h5')

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
        Nm_3fr_limit = 0

        while True:
            Nm_fr = Nm_fr + 1
            Nm_3fr_limit = Nm_3fr_limit + 1
            
            cls.IDvalue = cls.IDvalue + cls.Mem_IDvalue # Sum Memory Object ID
            
            ret, frame = cap.read()
            if not ret:
                break
            
            if Nm_3fr_limit > 3:
                Nm_3fr_limit = 0

                cls.Mem_IDvalue = cls.IDvalue # Memory Last Object ID

            image = Image.fromarray(frame)
            print(frame.shape)
            print("PROCESSING FRAME = ", Nm_fr)
            prediction = cls.detect_image(image)
            predictions.append(prediction)

            cls.IDvalue = 0 # Reset Object ID/frame

        cap.release()
        return {fname: predictions}
    
    @classmethod
    def pw_outdouga(cls, input):
        output_path = "../output/" + str(cls.IDvalue) + "result_douga.mp4"

        cap = cv2.VideoCapture(input)
        fname = os.path.basename(input)

        video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        print("video_FourCC = " , video_FourCC)#1983148141
        print("video_fps = " , video_fps)  # 5.0
        print("video_size = " , video_size) # (1936, 1216)
        # output
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        
        Nm_fr = 0

        while True:
            Nm_fr = Nm_fr + 1

            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(frame)
            print(frame.shape)
            print("PROCESSING FRAME = ", Nm_fr)
            image = cls.ret_frame(image)# ここでフレーム毎＝画像イメージ毎に動画をバラしている
            result = np.asarray(image)

            cv2.putText(result, text=str(cls.IDvalue), org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)

            if ret == True:
                out.write(result)

        cap.release()

    @classmethod
    def _get_class(cls, model_path='../src'):
        classes_path = os.path.join(model_path, 'voc_10classes.txt')
      
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    @classmethod
    def _get_anchors(cls, model_path='../src'):
        anchors_path = os.path.join(model_path, '2020_yolo_anchors9_trainallimages.txt')
      
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    @classmethod
    def compute_output(cls, image_data, image_shape):
        input_image_shape = tf.constant(image_shape)

        class_names = cls._get_class()
        anchors = cls._get_anchors()
        iou = 0.2    #Adjust param
        score = 0.8   #Adjust param
      
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
                #Car_result = {'id': int(0), 'box2d': [int(0),int(0),int(image.height),int(image.width)]}
                #Pedestrian_result = {'id': int(0), 'box2d': [int(0),int(0),int(image.height),int(image.width)]}

                if predicted_class == 'Car':
                    cls.IDvalue = cls.IDvalue + 1
                    #車を検出した時
                    Car_result = {'id': int(cls.IDvalue), 'box2d': [left,top,right,bottom]}#予測結果

                    #検出したオブジェクトを格納 検出しない場合は初期値０が格納される
                    Car_result_ALL.append(Car_result)#車
                    #Pedestrian_result_ALL.append(Pedestrian_result)#歩行者
                  
                elif predicted_class == 'Pedestrian':
                    cls.IDvalue = cls.IDvalue + 1
                    #歩行者を検出した時
                    Pedestrian_result = {'id': int(cls.IDvalue), 'box2d': [left,top,right,bottom]}#予測結果
              
                    #検出したオブジェクトを格納 検出しない場合は初期値０が格納される
                    #Car_result_ALL.append(Car_result)#車
                    Pedestrian_result_ALL.append(Pedestrian_result)#歩行者
        
        all_result = {'Car': Car_result_ALL, 'Pedestrian': Pedestrian_result_ALL}
        end = timer()
        print("1フレームの処理時間 = ", end - start)
        return all_result

    @classmethod
    def ret_frame(cls, image):
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / 10, 1., 1.) for x in range(10)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

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
        
        font = ImageFont.truetype(font='../box_font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            
            label = '{}_{:.2f}_{}'.format(predicted_class, score, str(cls.IDvalue))#put the ID for each obj
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

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

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
         
            #1 予測結果より次のFrameの物体位置を予測
            #nxt_result_txt = ' {},{},{},{},{}'.format(left, top, right, bottom, c)
            
            #2 検出したbox_sizeを計算する　設定した閾値1024pix**2
            sq_bdbox = (bottom - top)*(right - left) 

            if sq_bdbox >= 1024:#矩形サイズの閾値
                if predicted_class == 'Car'or predicted_class == 'Pedestrian':# Car or Pedes
                    # My kingdom for a good redistributable image drawing library.
                    for i in range(thickness):
                        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
                    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
                    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                    del draw
        
        end = timer()
        print("1フレームの処理時間 = ", end - start)
        return image

    

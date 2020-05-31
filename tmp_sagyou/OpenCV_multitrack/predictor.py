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

#from m1_tracker import Tracker

class ScoringService(object):
    IDvalue = 0 # クラス変数を宣言 = 0
    Mem_IDvalue = 0 # クラス変数を宣言 = 0
    
    #前フレームで検出した結果を格納
    old_out_boxes = []
    old_out_scores = []
    old_out_classes = []
    all_ObjectID_pos = []
    #tracker = Tracker(150, 30, 5)
    
    trackers = cv2.MultiTracker_create()#Multi Object tracker init

    @classmethod
    def get_model(cls, model_path='../model'):
        cls.IDvalue = 0 # Reset Object ID
        cls.Mem_IDvalue = 0 # Reset Memory Object ID
        cls.trackers = cv2.MultiTracker_create()#Multi Object tracker init
        
        modelpath = os.path.join(model_path, 'YOLOv3_608_cl10_val_loss71.h5')

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
            
            #cls.IDvalue = cls.Mem_IDvalue # Sum Memory Object ID
            
            ret, frame = cap.read()
            if not ret:
                break
            
            if Nm_3fr_limit > 11:
                Nm_3fr_limit = 0

                cls.Mem_IDvalue = cls.IDvalue # Memory Last Object ID

            image = Image.fromarray(frame)
            print(frame.shape)
            print("PROCESSING FRAME = ", Nm_fr)
            prediction = cls.detect_image(image)
            predictions.append(prediction)

            #cls.IDvalue = 0 # Reset Object ID/frame

        cap.release()
        return {fname: predictions}
    
    @classmethod
    def pw_outdouga(cls, input):
        cap = cv2.VideoCapture(input)
        fname = os.path.basename(input)

        output_path = "../output/" + "result_"+ fname
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
            image = cls.ret_frame(image, frame, Nm_fr)# ここでフレーム毎＝画像イメージ毎に動画をバラしている
            result = np.asarray(image)

            #cv2.putText(result, text=str(cls.IDvalue), org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #        fontScale=0.50, color=(255, 0, 0), thickness=2)

            if ret == True:
                out.write(result)

        cap.release()
    
    @classmethod
    def ret_frame(cls, image, cv2image, frame_num):
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / 10, 1., 1.) for x in range(10)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        start = timer()
      
        model_image_size = (608, 608)
        class_names = cls._get_class()

        new_image_size = (image.width - (image.width % 32),
                                image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
      
        image_shape = [image.size[1], image.size[0]]

        out_boxes, out_scores, out_classes = cls.compute_output(image_data, image_shape)
        
        if frame_num == 1:#
            cls.old_out_boxes = out_boxes
            cls.old_out_scores = out_scores
            cls.old_out_classes = out_classes
            backward_out_boxes = cls.old_out_boxes
            backward_out_scores = cls.old_out_scores
            backward_out_classes = cls.old_out_classes
        else:
            backward_out_boxes = cls.old_out_boxes
            backward_out_scores = cls.old_out_scores
            backward_out_classes = cls.old_out_classes
            #cls.old_out_boxes = 0 #クリアする
            #cls.old_out_scores = 0 #クリアする
            #cls.old_out_classes = 0 #クリアする
            cls.old_out_boxes = out_boxes#新しい検出結果に更新する
            cls.old_out_scores = out_scores#新しい検出結果に更新する
            cls.old_out_classes = out_classes#新しい検出結果に更新する

        current_pos = []

        font = ImageFont.truetype(font='../box_font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        thickness = (image.size[0] + image.size[1]) // 300
        
        #Check new object or not
        for it, ct in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[ct]
            box = out_boxes[it]
            score = out_scores[it]
            
            print("box = ", box)
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            
            boxcent_xpos = int((bottom - top)/2)
            boxcent_ypos = int((right - left)/2)
            
            if frame_num == 1:#1フレーム目は全て追加
                cls.IDvalue = cls.IDvalue + 1#全てObject IDを付与
                tmp = {'ID':cls.IDvalue, 'box_ord':it, 'x':boxcent_xpos, 'y':boxcent_ypos}
                current_pos.append(tmp)
                #LOGGING
                tmp = {'FRNUM':frame_num, 'ID':cls.IDvalue, 'box_ord':it, 'x':boxcent_xpos, 'y':boxcent_ypos}
                cls.all_ObjectID_pos.append(tmp)
                print("No.1 frame box center = ", current_pos)
                
            else:#それ以外前のフレームとの差分で新しい objet IDがあるかチェックする
                #cls.old_out_boxes
                #cls.old_out_scores
                #cls.old_out_classes
                
                for iold, cold in reversed(list(enumerate(cls.old_out_classes))):
                    predicted_class_old = class_names[cold]
                    box_old = out_boxes[iold]
                    score_old = out_scores[iold]
                    
                    top_old, left_old, bottom_old, right_old = box_old
                    top_old = max(0, np.floor(top_old + 0.5).astype('int32'))
                    left_old = max(0, np.floor(left_old + 0.5).astype('int32'))
                    bottom_old = min(image.size[1], np.floor(bottom_old + 0.5).astype('int32'))
                    right_old = min(image.size[0], np.floor(right_old + 0.5).astype('int32'))

                    #今回検出結果が前フレームで検出したBOX範囲内かチェックする
                    if not top_old < boxcent_ypos < bottom_old:
                        if not left_old < boxcent_xpos < right_old:
                            cls.IDvalue = cls.IDvalue + 1
                            tmp = {'ID':cls.IDvalue, 'box_ord':it, 'x':boxcent_xpos, 'y':boxcent_ypos}
                            current_pos.append(tmp)
                            #LOGGING
                            tmp = {'FRNUM':frame_num, 'ID':cls.IDvalue, 'box_ord':it, 'x':boxcent_xpos, 'y':boxcent_ypos}
                            cls.all_ObjectID_pos.append(tmp)
                            print("New object in frame ::box center = ", current_pos)
                    
        #current_pos check
        print("current_pos = ", len(current_pos))#
        for kt in range(len(current_pos)):
            tmp_current_pos = current_pos[kt]
            for k, v in tmp_current_pos.items():
                # k= Tanaka v= 80 // Tanaka: 80
                if k == "ID":
                    print("Key = ", k)
                    print("Value = ",v)             
                elif k == "box_ord":
                    print("Key = ", k)
                    print("Value = ",v)             

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            
            label = '{}_{:.2f}_{}'.format(predicted_class, score, str(cls.IDvalue))#put the ID for each obj

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
            if len(current_pos) > 0:
                t_tuple = (left, top, int(right - left), int(bottom - top))
                bbox = t_tuple
                #tracker = cv2.TrackerMedianFlow_create()
                tracker = cv2.TrackerKCF_create()
                cls.trackers.add(tracker, cv2image, bbox)

            track, boxes = cls.trackers.update(cv2image)
            
            if track:#trackingに成功したら
                for bbox in boxes:
                    #(x, y, w, h) = [int(v) for v in box]
                    #IDvalue_track = 
                    top_track    = int(bbox[1])
                    left_track   = int(bbox[0])
                    bottom_track = int(bbox[1] + bbox[3])
                    right_track  = int(bbox[0] + bbox[2])
                    
                    chksq_bdbox = (bottom_track - top_track)*(right_track - left_track) 
                    if chksq_bdbox >= 1024:#矩形サイズの閾値
                        if predicted_class == 'Car'or predicted_class == 'Pedestrian':# Car or Pedes
                            draw = ImageDraw.Draw(image)
                            label_size = draw.textsize(label, font)
                            
                            if top - label_size[1] >= 0:
                                text_origin = np.array([left_track, top_track - label_size[1]])
                            else:
                                text_origin = np.array([left_track, top_track + 1])


                            for i in range(thickness):
                                draw.rectangle([left_track + i, top_track + i, right_track - i, bottom_track - i], outline=colors[9])
                            del draw
            #else:#trackingに失敗したら
                #
                #del draw

            #2 検出したbox_sizeを計算する　設定した閾値1024pix**2
            #sq_bdbox = (bottom - top)*(right - left) 

            #if sq_bdbox >= 1024:#矩形サイズの閾値
            #    if predicted_class == 'Car'or predicted_class == 'Pedestrian':# Car or Pedes
                    # My kingdom for a good redistributable image drawing library.
                    #for i in range(thickness):
                    #    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
                    #draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
                    #draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                    #del draw
        
        end = timer()
        print("1フレームの処理時間 = ", end - start)
        return image

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
        iou = 0.3    #Adjust param
        score = 0.8   #Adjust param
      
        boxes, scores, classes = yolo_eval(cls.yolo_model(image_data), anchors,
                                             len(class_names), input_image_shape,
                                             score_threshold=score, iou_threshold=iou)
        return boxes, scores, classes
    
    @classmethod
    def detect_image(cls, image):
        start = timer()
      
        model_image_size = (608, 608)
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
            
            #center = (int((bottom - top)/2), int((right - left)/2))
            #center = np.array([int((bottom - top)/2), 1, int((right - left)/2)], dtype=np.int32)
            #cls.tracker.update(center)
            
            #for j in range(len(cls.tracker.tracks)):
            #    x = int(cls.tracker.tracks[j].trace[-1][0,0])
            #    y = int(cls.tracker.tracks[j].trace[-1][0,1])
            #    print("x=",x)
            #    print("y=",y)
            
            #2 検出したbox_sizeを計算する　設定した閾値1024pix**2
            sq_bdbox = (bottom - top)*(right - left) 

            if sq_bdbox >= 1024:#矩形サイズの閾値
                #検出しない時の初期値
                #Car_result = {'id': int(0), 'box2d': [int(0),int(0),int(image.height),int(image.width)]}
                #Pedestrian_result = {'id': int(0), 'box2d': [int(0),int(0),int(image.height),int(image.width)]}

                if predicted_class == 'Car':
                    #車を検出した時
                    Car_result = {'id': int(cls.IDvalue), 'box2d': [left,top,right,bottom]}#予測結果

                    #検出したオブジェクトを格納 検出しない場合は初期値０が格納される
                    Car_result_ALL.append(Car_result)#車

                elif predicted_class == 'Pedestrian':
                    #歩行者を検出した時
                    Pedestrian_result = {'id': int(cls.IDvalue), 'box2d': [left,top,right,bottom]}#予測結果
              
                    #検出したオブジェクトを格納 検出しない場合は初期値０が格納される
                    Pedestrian_result_ALL.append(Pedestrian_result)#歩行者
        
        all_result = {'Car': Car_result_ALL, 'Pedestrian': Pedestrian_result_ALL}
        end = timer()
        print("1フレームの処理時間 = ", end - start)
        return all_result

    

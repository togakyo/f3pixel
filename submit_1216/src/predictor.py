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
import copy

class ScoringService(object):
    IDvalue_car = 0 # Car クラス変数を宣言 = 0
    IDvalue_ped = 0 # Pedestrian クラス変数を宣言 = 0

    #前フレームで検出した結果を格納
    all_ObjectID_pos = []
    all_ObjectID_oldpos = []
    all_ObjectID_ped_pos = []
    all_ObjectID_ped_oldpos = []

    hit_oldpos = []

    @classmethod
    def get_model(cls, model_path='../model'):
        modelpath = os.path.join(model_path, 'tinyYOLOv3_1216_cl2_ep008_loss40.h5')

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

        #cls.IDvalue_car = 0 # Reset Object ID
        #cls.IDvalue_ped = 0 #

        #cls.all_ObjectID_pos = []
        #cls.all_ObjectID_oldpos = []

        #cls.all_ObjectID_ped_pos = []
        #cls.all_ObjectID_ped_oldpos = []

        predictions = []
        cap = cv2.VideoCapture(input)
        fname = os.path.basename(input)

        Nm_fr = 0

        while True:
            Nm_fr = Nm_fr + 1

            ret, frame = cap.read()
            if not ret:
                break

            im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(im_rgb)

            print(frame.shape)
            print("PROCESSING FRAME = ", Nm_fr)
            prediction, cls.IDvalue_car, cls.IDvalue_ped = cls.detect_image(image, Nm_fr, cls.IDvalue_car, cls.IDvalue_ped, cls.all_ObjectID_pos, cls.all_ObjectID_oldpos, cls.all_ObjectID_ped_pos, cls.all_ObjectID_ped_oldpos)

            predictions.append(prediction)

        cap.release()
        return {fname: predictions}

    @classmethod
    def detect_image(cls, Aimage, Aframe_num, AIDvalue_car, AIDvalue_ped, Aall_ObjectID_pos, Aall_ObjectID_oldpos, Aall_ObjectID_ped_pos, Aall_ObjectID_ped_oldpos):
        start = timer()

        class_names = cls._get_class()

        new_image_size = (Aimage.width - (Aimage.width % 32),
                                Aimage.height - (Aimage.height % 32))
        boxed_image = letterbox_image(Aimage, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        image_shape = [Aimage.size[1], Aimage.size[0]]

        out_boxes, out_scores, out_classes = cls.compute_output(image_data, image_shape)

        Car_result_ALL = []
        Pedestrian_result_ALL = []
        all_result = []

        # フレーム単位の処理
        if Aframe_num > 1:
            # Car / Pedestrian分ける
            Aall_ObjectID_oldpos = copy.copy(Aall_ObjectID_pos)
            Aall_ObjectID_pos.clear()

            Aall_ObjectID_ped_oldpos = copy.copy(Aall_ObjectID_ped_pos)
            Aall_ObjectID_ped_pos.clear()

        #オブジェクト単位の処理
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(Aimage.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(Aimage.size[0], np.floor(right + 0.5).astype('int32'))

            #JSON 形式の時はint32()未対応のため -> int()に変換する
            top = int(top)
            left = int(left)
            bottom = int(bottom)
            right = int(right)

            #2 検出したbox_sizeを計算する　設定した閾値1024pix**2
            sq_bdbox = (bottom - top)*(right - left)
            #3 検出したboxの中心点座標を計算する
            center_bdboxX = int((right - left)/2) + left
            center_bdboxY = int((bottom - top)/2) + top

            if sq_bdbox >= 1024:#矩形サイズの閾値
                if predicted_class == 'Car':
                    retobjID, ret_tmpcar = cls.switch_oldID(Aall_ObjectID_oldpos, AIDvalue_car, Aframe_num, center_bdboxX, center_bdboxY, left, top, right, bottom)
                    AIDvalue_car = retobjID# point
                    Aall_ObjectID_pos.append(ret_tmpcar)

                    Car_result = {'id': int(retobjID), 'box2d': [left,top,right,bottom]}#予測結果

                    #検出したオブジェクトを格納 検出しない場合は空欄が格納される
                    Car_result_ALL.append(Car_result)#車

                elif predicted_class == 'Pedestrian':
                    retobjID, ret_tmpped = cls.switch_oldID(Aall_ObjectID_ped_oldpos, AIDvalue_ped, Aframe_num, center_bdboxX, center_bdboxY, left, top, right, bottom)
                    AIDvalue_ped = retobjID# point
                    Aall_ObjectID_ped_pos.append(ret_tmpped)

                    Pedestrian_result = {'id': int(retobjID), 'box2d': [left,top,right,bottom]}#予測結果

                    #検出したオブジェクトを格納 検出しない場合は空欄が格納される
                    Pedestrian_result_ALL.append(Pedestrian_result)#歩行者

        all_result = {'Car': Car_result_ALL, 'Pedestrian': Pedestrian_result_ALL}
        end = timer()
        print("1フレームの処理時間 = ", end - start)
        # オブジェクトの検出結果と使用したIDを返す
        return all_result, AIDvalue_car, AIDvalue_ped

    @classmethod
    def pw_outdouga(cls, input):
        #cls.IDvalue_car = 0 # Reset Object ID
        #IDvalue_car = 0 # Reset Object ID

        #cls.all_ObjectID_pos = []
        #cls.all_ObjectID_oldpos = []

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

            im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(im_rgb)

            print("PROCESSING FRAME = ", Nm_fr)
            image, cls.IDvalue_car = cls.ret_frame(image, Nm_fr, cls.all_ObjectID_pos, cls.all_ObjectID_oldpos, cls.IDvalue_car)
            #image = cls.ret_frame(image, Nm_fr)
            # ここでフレーム毎＝画像イメージ毎に動画をバラしている

            result = np.asarray(image)
            im_rgbresult = result[:, :, [2, 1, 0]]
            if ret == True:
                out.write(im_rgbresult)

        cap.release()

    @classmethod
    def ret_frame(cls, image, frame_num, all_ObjectID_pos, all_ObjectID_oldpos, IDvalue):
        hsv_tuples = [(x / 10, 1., 1.) for x in range(10)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        start = timer()

        class_names = cls._get_class()

        new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        image_shape = [image.size[1], image.size[0]]

        out_boxes, out_scores, out_classes = cls.compute_output(image_data, image_shape)

        font = ImageFont.truetype(font='../box_font/FiraMono-Medium.otf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        # フレーム単位の処理
        if frame_num > 1:
            # Car / Pedestrian分ける
            all_ObjectID_oldpos = copy.copy(all_ObjectID_pos)
            all_ObjectID_pos.clear()

            cls.hit_oldpos.clear()#動画描画用

            #all_ObjectID_ped_oldpos = copy.copy(all_ObjectID_ped_pos)
            #all_ObjectID_ped_pos = []

        #print("out_classes = ", out_classes)
        # クラス分のfor文
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            #JSON 形式の時はint32()未対応のため -> int()に変換する
            top, left, bottom, right = box
            top    = int(max(0, np.floor(top + 0.5).astype('int32')))
            left   = int(max(0, np.floor(left + 0.5).astype('int32')))
            bottom = int(min(image.size[1], np.floor(bottom + 0.5).astype('int32')))
            right  = int(min(image.size[0], np.floor(right + 0.5).astype('int32')))

            #2 検出したbox_sizeを計算する　設定した閾値1024pix**2
            sq_bdbox = (bottom - top)*(right - left)
            #print("i = ",i)
            #print("c = ",c)

            if sq_bdbox >= 1024:#矩形サイズの閾値 1024
                #if predicted_class == 'Car'or predicted_class == 'Pedestrian':# Car or Pedes
                if predicted_class == 'Car':
                    #3 検出したboxの中心点座標を計算する
                    center_bdboxX = int((right - left)/2) + left
                    center_bdboxY = int((bottom - top)/2) + top

                    retobjID, ret_tmpcar = cls.switch_oldID(all_ObjectID_oldpos, IDvalue, frame_num, center_bdboxX, center_bdboxY, left, top, right, bottom)
                    IDvalue = retobjID#

                    print("Return ID = ", retobjID)
                    all_ObjectID_pos.append(ret_tmpcar)

                    label = '{}_{:.2f}_{}'.format(predicted_class, score, str(retobjID))#put the ID for each obj
                    draw = ImageDraw.Draw(image)
                    label_size = draw.textsize(label, font)
                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])

                    # My kingdom for a good redistributable image drawing library.
                    for i in range(thickness):
                        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[0])
                    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[0])
                    draw.text(text_origin, label, fill=(0, 0, 0), font=font)

                    if frame_num > 1 and (len(cls.hit_oldpos) != 0):
                        choose_old_pos = cls.hit_oldpos[0]
                        old_id         = cls.getValue('old_id',     choose_old_pos)
                        old_left       = cls.getValue('old_left',   choose_old_pos)
                        old_top        = cls.getValue('old_top',    choose_old_pos)
                        old_right      = cls.getValue('old_right',  choose_old_pos)
                        old_bottom     = cls.getValue('old_bottom', choose_old_pos)

                        label = '{}_{:.2f}_{}'.format(predicted_class, score, str(old_id))

                        if old_top - label_size[1] >= 0:
                            text_origin2 = np.array([old_left, old_top - label_size[1]])
                        else:
                            text_origin2 = np.array([old_left, old_top + 1])

                        for i in range(thickness):
                            draw.rectangle([old_left + i, old_top + i, old_right - i, old_bottom - i], outline=colors[1])
                        draw.rectangle([tuple(text_origin2), tuple(text_origin2 + label_size)], fill=colors[1])
                        draw.text(text_origin2, label, fill=(0, 0, 0), font=font)

                        del draw

        end = timer()
        print("1フレームの処理時間 = ", end - start)
        #オブジェクト検出した結果の画像データ、使用したIDを返す
        return image, IDvalue

    @classmethod
    def _get_class(cls, model_path='../src'):
        classes_path = os.path.join(model_path, 'voc_2classes.txt')

        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    @classmethod
    def _get_anchors(cls, model_path='../src'):
        anchors_path = os.path.join(model_path, '2020_yolo_anchors6_FOC2_allimg.txt')

        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    @classmethod
    def compute_output(cls, image_data, image_shape):
        input_image_shape = tf.constant(image_shape)

        class_names = cls._get_class()
        anchors = cls._get_anchors()
        iou = 0.4    #Adjust param
        score = 0.5   #Adjust param

        boxes, scores, classes = yolo_eval(cls.yolo_model(image_data), anchors,
                                             len(class_names), input_image_shape,
                                             score_threshold=score, iou_threshold=iou)
        return boxes, scores, classes

    @classmethod
    def getValue(cls, key, items):
        values = [x['Value'] for x in items if 'Key' in x and 'Value' in x and x['Key'] == key]
        return values[0] if values else None

    @classmethod
    def switch_oldID(cls, oldpos, currentID, frame_num, pos_centx, pos_centy, left, top, right, bottom):
        matches_cnt = 0
        tmpObjID_setimg = 0

        if frame_num == 1:#1フレーム目は全て登録する
            currentID = currentID + 1
            tmpObjID_setimg = currentID
        else:
            for kt in range(len(oldpos)):
                tmp_old_pos = oldpos[kt]

                old_left   = cls.getValue('left', tmp_old_pos)
                old_top    = cls.getValue('top', tmp_old_pos)
                old_right  = cls.getValue('right', tmp_old_pos)
                old_bottom = cls.getValue('bottom', tmp_old_pos)

                band_value     = 5 #Adjust param　検出する範囲を狭める
                exp_old_left   = int(old_left   + band_value)
                exp_old_top    = int(old_top    + band_value)
                exp_old_right  = int(old_right  - band_value)
                exp_old_bottom = int(old_bottom - band_value)

                if(( pos_centx >= exp_old_left ) and ( pos_centx <= exp_old_right )):
                    if(( pos_centy >= exp_old_top) and ( pos_centy <= exp_old_bottom )):

                        matches_cnt     = matches_cnt + 1
                        tmpObjID_setimg = cls.getValue('id', tmp_old_pos)

                        mem_oldpos = [{'Key':'frame',      'Value':frame_num},
                                      {'Key':'old_id',     'Value':tmpObjID_setimg},
                                      {'Key':'old_left',   'Value':old_left},
                                      {'Key':'old_top',    'Value':old_top},
                                      {'Key':'old_right',  'Value':old_right},
                                      {'Key':'old_bottom', 'Value':old_bottom}]
                        cls.hit_oldpos.append(mem_oldpos)

            #前回フレームより過去のオブジェクトを全てチェックした結果を出力
            print("matches_cnt = ", matches_cnt)

            #もしどのIDにも当てはまらない場合
            if matches_cnt == 0:
                currentID = currentID + 1
                tmpObjID_setimg = currentID
            #else:
                #nashi

        #更新したObjIDを登録する
        tmp_info = [{'Key':'frame',  'Value':frame_num},
                    {'Key':'id',     'Value':tmpObjID_setimg},
                    {'Key':'left',   'Value':left},
                    {'Key':'top',    'Value':top},
                    {'Key':'right',  'Value':right},
                    {'Key':'bottom', 'Value':bottom}]

        return tmpObjID_setimg, tmp_info

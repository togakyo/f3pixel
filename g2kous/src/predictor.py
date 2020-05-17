import glob
import os
import cv2
import pickle

# add by togakyo
import colorsys
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from tensorflow.keras.utils import multi_gpu_model


class ScoringService(object):
    _defaults = {
        "model_path": '../model/10class/tinyYOLOv3_cl10_val_loss21.h5',
        "anchors_path": '../model/10class/2020_yolo_cl10_anchors.txt',
        "classes_path": '../model/10class/voc_10classes.txt',
        "score": 0.3,
        "iou": 0.5,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }
    #model_path, anchors_path, classes_pathを変更すれば、１０クラス検出版と２クラス検出版に切替可
    
    @classmethod
    def get_defaults(cls, n):
      if n in cls._defaults:
          return cls._defaults[n]
      else:
          return "Unrecognized attribute name '" + n + "'"
    
    def __init__(self, **kwargs):
      self.__dict__.update(self._defaults)  # set up default values
      # self.__dict__.update(kwargs)  # and update with user overrides
      self.class_names = self._get_class()
      self.anchors = self._get_anchors()
      self.load_yolo_model()

    def _get_class(self):
      classes_path = os.path.expanduser(self.classes_path)
      print(classes_path)
      with open(classes_path) as f:
          class_names = f.readlines()
      class_names = [c.strip() for c in class_names]
      return class_names

    def _get_anchors(self):
      anchors_path = os.path.expanduser(self.anchors_path)
      with open(anchors_path) as f:
          anchors = f.readline()
      anchors = [float(x) for x in anchors.split(',')]
      return np.array(anchors).reshape(-1, 2)

    def load_yolo_model(self):
      model_path = os.path.expanduser(self.model_path)
      assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

      # Load model, or construct model and load weights.
      num_anchors = len(self.anchors)
      num_classes = len(self.class_names)
      is_tiny_version = num_anchors == 6  # default setting
      try:
          #self.yolo_model = load_model(model_path, compile=False)
          self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
              if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
          self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
          return True
      except:
          return False
            
      else:
          assert self.yolo_model.layers[-1].output_shape[-1] == \
                  num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
               'Mismatch between model and given anchor and class sizes'

      print('{} model, anchors, and classes loaded.'.format(model_path))


    def compute_output(self, image_data, image_shape):
      # Generate output tensor targets for filtered bounding boxes.
      # self.input_image_shape = K.placeholder(shape=(2,))
      self.input_image_shape = tf.constant(image_shape)
      if self.gpu_num >= 2:
        self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)

      boxes, scores, classes = yolo_eval(self.yolo_model(image_data), self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
      return boxes, scores, classes

    def detect_image(self, image):
      start = timer()

      if self.model_image_size != (None, None):
          assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
          assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
          boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
      else:
          new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
          boxed_image = letterbox_image(image, new_image_size)
      image_data = np.array(boxed_image, dtype='float32')

      image_data /= 255.
      image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

      out_boxes, out_scores, out_classes = self.compute_output(image_data, [image.size[1], image.size[0]])

      print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

      Car_result_ALL = []
      Pedestrian_result_ALL = []
      all_result = []
      for i, c in reversed(list(enumerate(out_classes))):
          predicted_class = self.class_names[c]
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
         
          print(label, (left, top), (right, bottom))# Frame毎の検出位置

          #1 予測結果より次のFrameの物体位置を予測
          nxt_result_txt = ' {},{},{},{},{}'.format(left, top, right, bottom, c)
            
          #2 検出したbox_sizeを計算する　設定した閾値1024pix**2
          sq_bdbox = (bottom - top)*(right - left) 
          print("矩形の大きさ:", sq_bdbox)

          if sq_bdbox >= 1024:#矩形サイズの閾値
              #検出しない時の初期値
              Car_result = {'id': int(0), 'box2d': [int(0),int(0),int(0),int(0)]}
              Pedestrian_result = {'id': int(0), 'box2d': [int(0),int(0),int(0),int(0)]}

              if predicted_class == 'Car':
                  #車を検出した時
                  Car_result = {'id': int(0), 'box2d': [left,top,right,bottom]}#予測結果
                  
              elif predicted_class == 'Pedestrian':
                  #歩行者を検出した時
                  Pedestrian_result = {'id': int(0), 'box2d': [left,top,right,bottom]}#予測結果
              
              #検出したオブジェクトを格納 検出しない場合は初期値０が格納される
              Car_result_ALL.append(Car_result)#車
              Pedestrian_result_ALL.append(Pedestrian_result)#歩行者
        
      all_result = {'Car': Car_result_ALL, 'Pedestrian': Pedestrian_result_ALL}
      end = timer()
      print("1フレームの処理時間 = ", end - start)
      return all_result

    def predict(self, input):
      predictions = []
      cap = cv2.VideoCapture(input)
      fname = os.path.basename(input)

      while True:
        ret, frame = cap.read()
        if not ret:
          break
            
        image = Image.fromarray(frame)
        prediction = self.detect_image(image)

        predictions.append(prediction)
      cap.release()
      return {fname: predictions}

import glob
import os
import cv2
import pickle

class ScoringService(object):
    @classmethod
    def get_model(cls, model_path='../model'):
        """Get model method
 
        Args:
            model_path (str): Path to the trained model directory.
 
        Returns:
            bool: The return value. True for success, False otherwise.
        
        Note:
            - You cannot connect to external network during the prediction,
              so do not include such process as using urllib.request.
 
        """
        model_files = glob.glob(model_path+'/*.pkl')
        try:
            if len(model_files):
                model_file = model_files[0]
                with open(os.path.join(model_path, model_file), 'rb') as f:
                    cls.model = pickle.load(f)
            else:
                cls.model = None
            
            return True
        
        except:
            return False

    @classmethod
    def predict(cls, input):
        """Predict method
 
        Args:
            input (str): path to the video file you want to make inference from
 
        Returns:
            dict: Inference for the given input.
                format:
                    - filename []:
                        - category_1 []:
                            - id: int
                            - box2d: [left, top, right, bottom]
                        ...
        Notes:
            - The categories for testing are "Car" and "Pedestrian".
              Do not include other categories in the prediction you will make.
            - If you do not want to make any prediction in some frames,
              just write "prediction = {}" in the prediction of the frame in the sequence(in line 65 or line 67).
        """
        predictions = []
        cap = cv2.VideoCapture(input)
        fname = os.path.basename(input)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if cls.model is not None:
                prediction = cls.model.predict(frame)
            else:
                prediction = {"Car": [{"id": 0, "box2d": [0, 0, frame.shape[1], frame.shape[0]]}],
                              "Pedestrian": [{"id": 0, "box2d": [0, 0, frame.shape[1], frame.shape[0]]}]}
            predictions.append(prediction)
        cap.release()
        
        return {fname: predictions}

import sys
from PIL import Image
import json
import glob
import cv2
import os

if __name__ == '__main__':
    #FLAGS = parser.parse_args()
    data_path = 'data'#読みこむデータのパスを記載
    videos = sorted(glob.glob(data_path+'/*.mp4'))
    print("ALL mp4 = ", videos)
    
    #フレームのファイル名を動画をまたいで連番で付与
    Number_frseries = 0
    
    for i in range(len(videos)):
        video_path = videos[i]
    
        cap = cv2.VideoCapture(video_path)
        fname = os.path.basename(video_path)
        
        video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        save_mp4path = 'out/result_{}.mp4'.format(fname)
        out = cv2.VideoWriter(save_mp4path, video_FourCC, video_fps, video_size)

        while True:
            ret, frame = cap.read()
            
            Number_frseries = Number_frseries + 1
            
            if not ret:
                break
            
            #image = Image.fromarray(frame)
            #print(image.mode)
            # RGB
            save_imgpath = 'out/{}.jpg'.format(Number_frseries)
            #save_imgpath = 'out_frame/{}.bmp'.format(Number_frseries)
            
            yAxis = cv2.flip(frame, 1)
            
            #image.save(save_imgpath)
            #cv2.imwrite(save_imgpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            #cv2.imwrite(save_imgpath, frame)
            cv2.imwrite(save_imgpath, yAxis, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            out.write(yAxis)
            print("Save_OK Frame = ", save_imgpath)
        print("＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊")

    #with open('../output/prediction.json', 'w') as f:
    #    json.dump(Output_list, f)


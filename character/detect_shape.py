import math
import cv2
import mediapipe as mp
import os
import numpy as np
import math
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def angle(p1, p2):
    angle1 = np.arctan2(*p1[::-1])
    angle2 = np.arctan2(*p2[::-1])
    result = np.rad2deg((angle1-angle2) % (2*np.pi))
    return result

def getAngle(p1,p2,p3, direction='CW'):
    pt1 = (p1[0] - p2[0], p1[1] - p2[1])
    pt2 = (p3[0] - p2[0], p3[1] - p2[1])
    result = angle(pt1, pt2)
    result = (result + 360) % 360
    if direction == "CCW":    #반시계방향
        result = (360 - result) % 360
    return result

def distance(p1, p2):
    result = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return result

def extract_facedata(img):
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(static_image_mode=True,
                                max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5) as face_mesh:
            image = img
            # image = cv2.imread(img)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                print('Undetected')
                return 'Undetected'
            # annotated_image = image.copy()
            mesh_dot = results.multi_face_landmarks[0].landmark
            top_x, top_y = mesh_dot[10].x , mesh_dot[10].y
        
            l_top_x, l_top_y = mesh_dot[284].x , mesh_dot[284].y
            l_mid_x, l_mid_y = mesh_dot[454].x , mesh_dot[454].y
            l_bot_x, l_bot_y = mesh_dot[397].x , mesh_dot[397].y
            l_bot_x2, l_bot_y2 = mesh_dot[379].x , mesh_dot[379].y
            l_bot_x3, l_bot_y3 = mesh_dot[400].x , mesh_dot[400].y
            
            r_top_x, r_top_y = mesh_dot[54].x , mesh_dot[54].x
            r_mid_x, r_mid_y = mesh_dot[234].x , mesh_dot[234].y
            r_bot_x, r_bot_y = mesh_dot[172].x , mesh_dot[172].y
            r_bot_x2, r_bot_y2 = mesh_dot[150].x , mesh_dot[150].y
            r_bot_x3, r_bot_y3 = mesh_dot[176].x , mesh_dot[176].y
            
            bot_x, bot_y = mesh_dot[152].x , mesh_dot[152].y
            
            top_xy = (int(image.shape[1]*top_x),int(image.shape[0]*top_y))
            
            # l_top_xy = (int(image.shape[1]*l_top_x),int(image.shape[0]*l_top_y))
            l_mid_xy = (int(image.shape[1]*l_mid_x),int(image.shape[0]*l_mid_y))
            l_bot_xy = (int(image.shape[1]*l_bot_x),int(image.shape[0]*l_bot_y))
            l_bot_xy2 = (int(image.shape[1]*l_bot_x2),int(image.shape[0]*l_bot_y2))
            l_bot_xy3 = (int(image.shape[1]*l_bot_x3),int(image.shape[0]*l_bot_y3))
            
            # r_top_xy = (int(image.shape[1]*r_top_x),int(image.shape[0]*r_top_y))
            r_mid_xy = (int(image.shape[1]*r_mid_x),int(image.shape[0]*r_mid_y))
            r_bot_xy = (int(image.shape[1]*r_bot_x),int(image.shape[0]*r_bot_y))
            r_bot_xy2 = (int(image.shape[1]*r_bot_x2),int(image.shape[0]*r_bot_y2))
            r_bot_xy3 = (int(image.shape[1]*r_bot_x3),int(image.shape[0]*r_bot_y3))
            
            bot_xy = (int(image.shape[1]*bot_x),int(image.shape[0]*bot_y))
            
            
            d1 = distance((l_mid_x,l_mid_y), (r_mid_x, r_mid_y)) # width
            d2 = distance((l_top_x, l_top_y), (r_top_x,r_top_y))
            d3 = distance((bot_x,bot_y), (top_x,top_y)) # height
            d4_l = distance((bot_x,bot_y), (l_bot_x,l_bot_y))
            d4_r = distance((bot_x,bot_y), (r_bot_x,r_bot_y))
            d5 = distance((l_bot_x,l_bot_y), (r_bot_x,r_bot_y))
            d6 = distance((l_bot_x2,l_bot_y2), (r_bot_x2,r_bot_y2))
            d7 = distance((l_bot_x3,l_bot_y3), (r_bot_x3,r_bot_y3))
            a1R = getAngle(top_xy,bot_xy,r_bot_xy)
            a1L = getAngle(top_xy,bot_xy,l_bot_xy,"CCW")
            a2R = getAngle(top_xy,bot_xy,r_bot_xy2)
            a2L = getAngle(top_xy,bot_xy,l_bot_xy2,"CCW")
            a3R = getAngle(top_xy,bot_xy,r_bot_xy3)
            a3L = getAngle(top_xy,bot_xy,l_bot_xy3,"CCW")
            a4R = getAngle(l_mid_xy,r_mid_xy,r_bot_xy,"CCW")
            a4L = getAngle(r_mid_xy,l_mid_xy,l_bot_xy)
            a5R = getAngle(r_mid_xy, r_bot_xy, r_bot_xy2,"CCW")
            a5L = getAngle(l_mid_xy, l_bot_xy, l_bot_xy2)
            a6R = getAngle(r_bot_xy, r_bot_xy2, r_bot_xy3,"CCW")
            a6L = getAngle(l_bot_xy, l_bot_xy2, l_bot_xy3)
            a7R = getAngle(r_bot_xy2, r_bot_xy3, bot_xy,"CCW")
            a7L = getAngle(l_bot_xy2, l_bot_xy3, bot_xy)
            left_chin = getAngle(l_mid_xy,l_bot_xy,bot_xy)
            right_chin = getAngle(r_mid_xy,r_bot_xy,bot_xy,"CCW")
            center_chin = getAngle(r_bot_xy3,bot_xy,l_bot_xy3,"CCW")
            
            data = np.array([left_chin, center_chin, right_chin, 
                            a1R,a1L, a2R,a2L, a3R,a3L, a4R,a4L, a5R,a5L, a6R,a6L, a7R,a7L,
                            d2/d1, d1/d3, d2/d3, d1/d5, d6/d5, d4_l/d6, d4_r/d6, d6/d1, d5/d2, d4_l/d5, d4_r/d6, d7/d6])
            return data
        
def face_classifi(image):
    shape_class_dict={'Oblong':'Square',
                  'Round':'Circle',
                  'Oval':'Oval',
                  'Square':'Triangle'}
    scalers = joblib.load('character/scalers.pkl')
    img = extract_facedata(image)
    img = img.reshape(1,-1)
    img = scalers.transform(img)
    svm_load_model = joblib.load('character/rbf_face_class_v4.pkl')
    pred = svm_load_model.predict(img)
    final_result = shape_class_dict[pred[0]]
    return final_result

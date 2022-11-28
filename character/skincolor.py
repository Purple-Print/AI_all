import mediapipe as mp
import cv2
import numpy as np


def select_color(R,G,B):         
    mR = np.mean(R)
    mG = np.mean(G)
    mB = np.mean(B)

    color = np.array([mR,mG,mB])
    preset1 = np.array([147.,90.,78.])
    preset2 = np.array([172.,136.,126.])
    preset3 = np.array([172.,143.,122.])
    preset4 = np.array([104.,63.,46.])

    color_norm = np.sqrt(np.sum(np.square(color)))
    preset1_norm = np.sqrt(np.sum(np.square(preset1)))
    preset2_norm = np.sqrt(np.sum(np.square(preset2)))
    preset3_norm = np.sqrt(np.sum(np.square(preset3)))
    preset4_norm = np.sqrt(np.sum(np.square(preset4)))

    similar_color = {
        'color1': np.sum(color*preset1)/(color_norm*preset1_norm),
        'color2': np.sum(color*preset2)/(color_norm*preset2_norm),
        'color3': np.sum(color*preset3)/(color_norm*preset3_norm),
        'color4': np.sum(color*preset4)/(color_norm*preset4_norm)

    }
    # print(similar_color)
    return max(similar_color, key = similar_color.get)

def skin_detect(img):
    contour_list = [152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377]
    point_list=[]
    B = []
    G = []
    R = []
    
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                                max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5) as face_mesh:
        image = img
                        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            print('Undetected')
            return 'Undetected'

        mesh_dot = results.multi_face_landmarks[0].landmark
        for i in contour_list:
            point_x = int(image.shape[1]*mesh_dot[i].x)
            point_y = int(image.shape[0]*mesh_dot[i].y)
            point_list.append([point_x,point_y])

            for i in point_list:
                colors = image[i[1]][i[0]]
                colors = colors.tolist()
                B.append(colors[0])
                G.append(colors[1])
                R.append(colors[2])
                
    result = select_color(R,G,B)
    return result
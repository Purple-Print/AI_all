
from heatmappy import Heatmapper
from PIL import Image
import numpy as np
import pandas as pd
import io
import cv2
import base64

def point_change(x,z):
    x = abs(x + 21.8) * 12.53
    z = abs(z -57) * 11.6
    point_list=(x,z)
    return point_list

def combine_2rd_columns(col_1, col_2):
    # result = int(col_1)
    # if not pd.isna(col_2):
    #     result += "," + str(col_2)
    if not pd.isna(col_2):
        result = (float(col_1),float(col_2))
    return result

def get_user_data(user_data):
    json_data = user_data.dict()
    user_data_list = json_data['coorperate']
    user_data_df = pd.DataFrame(user_data_list)
    user_data_df.sort_values('time')
    user_data_df["coord"] = user_data_df.apply(lambda x: combine_2rd_columns(x['x'], x['z']), axis=1)
    user_data_df = user_data_df.drop(['x', 'z'], axis=1)
    user_data_df = user_data_df.drop_duplicates(['id','time'])
    users = user_data_df.id.unique()
    log_Users = []
    for i in users:
        index_Dict = {}
        globals()['userid_{}'.format(i)] = user_data_df[user_data_df['id'] == '{}'.format(i)].drop(columns = "id",axis=1)
        globals()['userids_{}'.format(i)] = globals()['userid_{}'.format(i)]["coord"]
        time_Index = list(globals()['userid_{}'.format(i)]["time"])
        globals()['userid_{}'.format(i)] = pd.Series(globals()['userids_{}'.format(i)].values,index=time_Index).rename('{}'.format(i))
        log_Users.append('{}'.format(i)) 
    log_Table = pd.concat([globals()['userid_{}'.format(i)] for i in log_Users], axis=1)
    log_Table.fillna('Offline',inplace=True)
    # print(log_Table)
    return log_Table

def heatmap_maker(data):
    coord_df = get_user_data(data)
    coord_df = coord_df.replace("Offline",np.NaN)
    user_columns = coord_df.columns
    print(user_columns)
    img_path = 'uer_heatmap/map.png'
    img = Image.open(img_path)
    result=[]
    for id in user_columns:
        print(id)
        point_list=[]
        coord_list = coord_df[id].values
        print(coord_list)
        coord_list = [item for item in coord_list if not(pd.isnull(item)) == True]
        if len(coord_list) != 0:
            for xz in coord_list:
                point_list.append(point_change(xz[0],xz[1]))
                example_points = point_list
                heatmapper = Heatmapper(
                    point_diameter=200,
                    point_strength=0.7,
                    opacity=0.6,
                    colours='default',
                    grey_heatmapper='PIL'
                )
                heatmap = heatmapper.heatmap_on_img(example_points,img)
                buff = io.BytesIO()
                heatmap.save(buff,"PNG")
                imgByte=buff.getvalue()
                
                encoded = base64.b64encode(imgByte)
                imgByte = encoded.decode('ascii')
   
            heatmap_result = {
                'user_id':id,
                'byte_image':imgByte
            }
            result.append(heatmap_result)
        else:
            return 'no coordinate data'
    json_data = {"result":result}
    return json_data
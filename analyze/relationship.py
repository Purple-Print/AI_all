import pandas as pd
from character.detect_shape import distance
import numpy as np

def combine_2rd_columns(col_1, col_2):
    # result = int(col_1)
    # if not pd.isna(col_2):
    #     result += "," + str(col_2)
    if not pd.isna(col_2):
        result = [float(col_1),float(col_2)]
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
    return log_Table

def user_relationship(user_data):
    result = []
    log_df = get_user_data(user_data)
    user_columns = log_df.columns
    time_index_list = log_df.index
    friendly_df = pd.DataFrame(columns=user_columns, index=time_index_list)
    secondly_df = pd.DataFrame(columns=user_columns, index=time_index_list)
    # print(result_df)
    for time_recorded in time_index_list:
        nearly_user_name=''
        second_user_name=''
        base_user_name=''
        selected_df = log_df.loc[[time_recorded]]
        # print(selected_df) 
        for target_user in range(len(user_columns)):
            nearly_user_distance = 999
            second_user_distance = 999
            for compare_user in range(len(user_columns)):
                if user_columns[target_user] == user_columns[compare_user]:   # 같은 유저일 경우 패스
                    pass
                else: #다른 유저끼리 비교 
                    target, compare = user_columns[target_user], user_columns[compare_user]
                    target_data = selected_df[target].values[0]
                    compare_data = selected_df[compare].values[0]
                    if target_data !="Offline" and compare_data != "Offline":
                        between_user_dis = distance(target_data, compare_data)
                        if between_user_dis < nearly_user_distance :
                            second_user_distance = nearly_user_distance
                            nearly_user_distance = between_user_dis
                            base_user_name =target
                            second_user_name = nearly_user_name
                            nearly_user_name=compare
                        elif second_user_distance > between_user_dis:
                            second_user_distance=between_user_dis
                            base_user_name =target
                            second_user_name=compare

            if nearly_user_distance != 999:     
                friendly_df.loc[time_recorded,base_user_name]=nearly_user_name
                secondly_df.loc[time_recorded,base_user_name]=second_user_name
    print(friendly_df)
    print(secondly_df)
    location_df = user_location(user_data)
    for id in user_columns:
        friendly_list = friendly_df[id].values
        friendly_list = [item for item in friendly_list if not(pd.isnull(item)) == True]
        secondly_list = secondly_df[id].values
        secondly_list = [item for item in secondly_list if not(pd.isnull(item)) == True]
        if len(friendly_list)!= 0:
            most_friendly = max(friendly_list, key = friendly_list.count)
            spend_time_with_most = friendly_df[id].value_counts().max() * 10
            if most_friendly == "":
                most_friendly="None"
                spend_time_with_most=0
        else:
            most_friendly = "None"
            spend_time_with_most = 0
        
        if len(secondly_list) != 0:
            second_friendly = max(secondly_list, key = secondly_list.count)
            spend_time_with_second = secondly_df[id].value_counts().max() * 10
            if second_friendly=="":
                second_friendly="None"
                spend_time_with_second=0
        else:
            second_friendly = "None"
            spend_time_with_second = 0
            
        place_list = location_df[id].values
        place_list = [item for item in place_list if not(pd.isnull(item)) == True]
        most_place = max(place_list, key = place_list.count)  #가장 많이 논 곳
        place_time = location_df[id].value_counts().max() * 10  #가장 많이 논 곳 에서 보낸 시간초단위 
        user_info = {
                     'childid':id,
                     "friends":[
                         {"bestfriend_1":most_friendly,
                          "time_1":int(spend_time_with_most),
                          "bestfriend_2":second_friendly,
                          "time_2":int(spend_time_with_second)
                          }
                         ],
                     "place":most_place,
                     "spend time":int(place_time)
                     }
        result.append(user_info)
    json_data = {"result":result}
    return json_data

# 1. 놀이터 학교 장소 df 구하기
# 학교 : x:-21.8 ~ 20.8      z: 7.92 ~ 57
# 놀이터 : x:-18.07 ~ 12.55  z: -19.17 ~ 7.57

# z값이 -19.17~7.57//7.92~57
def user_location(user_data):
    log_df = get_user_data(user_data)
    user_columns = log_df.columns
    time_index_list = log_df.index
    location_df = pd.DataFrame(columns=user_columns, index=time_index_list)
    for time_recorded in time_index_list:
        selected_df = log_df.loc[[time_recorded]]
        for target in user_columns:
            target_data = selected_df[target].values[0]
            if target_data != 'Offline':
                if target_data[1] <= 7.57 :
                    target_data='놀이터'  
                else :
                    target_data='학교'
            location_df.loc[time_recorded,target]=target_data
    location_df = location_df.replace("Offline",np.NaN)
    # for id_location in user_columns:
    #     place_list = location_df[id_location].values
    #     place_list = [item for item in place_list if not(pd.isnull(item)) == True]
    #     most_place = max(place_list, key = place_list.count)  #가장 많이 논 곳
    #     place_time = location_df[id_location].value_counts().max() * 10  #가장 많이 논 곳 에서 보낸 시간초단위 
    return location_df
    
# 2. 접속시간

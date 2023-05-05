import pandas as pd
import numpy as np
import uuid
# Import Snowpark functions
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import col, lag
from snowflake.snowpark.window import Window
import streamlit as st


def bandit_data_set_grundstruktur(bandit_dataset, placeID):
    bandit_dataset['generatedID'] = [uuid.uuid4() for _ in range(len(bandit_dataset))]
    bandit_dataset['original_placeID'] = bandit_dataset[placeID]
    bandit_dataset['Action'] =bandit_dataset[placeID] 
    bandit_dataset['Action_nr'] =bandit_dataset[placeID] 
    bandit_dataset['prob_action'] = np.random.rand(len(bandit_dataset))
    new_id_df = pd.DataFrame({placeID: bandit_dataset[placeID].unique(),
                            'new_ID': range(1, 113)})
    merged_df = pd.merge(bandit_dataset, new_id_df, on=placeID)
    bandit_dataset = merged_df
    bandit_dataset['Action'] = bandit_dataset['new_ID']
    return bandit_dataset

def get_place_id_Mapping(bandit_dataset, placeID):
    bandit_dataset['generatedID'] = [uuid.uuid4() for _ in range(len(bandit_dataset))]
    bandit_dataset['original_placeID'] = bandit_dataset[placeID]

    return bandit_dataset
def placeIDgeben(data, PLACEID, Nummer):
    bandit_data = bandit_data_set_grundstruktur(data, PLACEID)
    bandit_ = bandit_data[['Action', 'PLACEID']]
    bandit_= bandit_[bandit_['Action'] == Nummer]
    return bandit_['PLACEID'].iloc[0]


def number_check(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    

def vowpal_wabit_context_string(user_feature_dict):
    feature_list = []
    for k, v in user_feature_dict.items():
        if number_check(v):
            feature_list.append(f"{k}:{v}")
        else:
            feature_list.append(f"{k}_{v}")
    return [f"{' '.join(feature_list)}"]



account = st.secrets.snowflake["account"] 
username = st.secrets.snowflake["user"] 
password = st.secrets.snowflake["password"] 
database = st.secrets.snowflake["database"] 


def create_snowflake_connection():
    """Create Snowpark session object"""
    connection_parameters = st.secrets["snowflake"]
    conn = Session.builder.configs(connection_parameters).create()
    return conn






def get_place_data(place_id):
    conn = create_snowflake_connection()

    try:
        query = f"""
        SELECT PLACEID, NAME, ADVERTISMENT, IMAGE_URL
        FROM restaurants
        WHERE PLACEID = %s;
        """

        cur = conn.cursor()
        cur.execute(query, (place_id,))
        result = cur.fetchall()
        columns = ['PlaceID', 'Name', 'Advertisment', 'Image_url']
        df = pd.DataFrame(result, columns=columns)
        return df

    finally:
        conn.close()


def training_cb(bandit, vowpal_workspace, filename):
    for i in bandit.generatedID.unique():
        bandit_loop = bandit[bandit['generatedID'] == i]
        context = bandit_loop[['VALIDATED_PARKING',  'MON_TUE_WED_THU_FRI', 'SAT', 'SUN','YES', 'BAKERY', 'BAR', 'BAR_PUB_BREWERY', 'BARBECUE', 'BREAKFAST_BRUNCH', 'MEXICAN', 'VEGETARIAN', 'INTERNATIONAL', ]].iloc[:1]
        context_dict = context.to_dict('r')
        bandit_loop.reset_index(inplace=True, drop=True)
        prob_action = bandit_loop.prob_action
        reward = bandit_loop.RATING * -1
        daten =[
        vowpal_wabit_context_string(context_dict[0])[0],
            

        ]
        daten = f"{bandit_loop.Action.iloc[0 ]}:{reward.iloc[0]}:{prob_action.iloc[0 ]} | {str(daten[0])[1:-1]}" 

        vowpal_workspace.learn(daten)
    vowpal_workspace.save(f"cb.{filename}")
    vowpal_workspace.finish()
    return print(daten)



def recommending_cb(bandit, vowpal_workspace):
    prediction_result = []

    for i in bandit.generatedID.unique():
        bandit_loop = bandit[bandit['generatedID'] == i]
        context = bandit_loop[['VALIDATED_PARKING',  'MON_TUE_WED_THU_FRI', 'SAT', 'SUN', 'YES', 'BAKERY', 'BAR', 'BAR_PUB_BREWERY', 'BARBECUE', 'BREAKFAST_BRUNCH', 'MEXICAN', 'VEGETARIAN', 'INTERNATIONAL', ]].iloc[:1]
        context_dict = context.to_dict('r')
        bandit_loop.reset_index(inplace=True, drop=True)

        daten =[
        vowpal_wabit_context_string(context_dict[0])[0],
            

        ]
        
        #daten = f"{bandit_loop.Action.iloc[0 ]}:{reward.iloc[0]}:{prob_action.iloc[0 ]} | {str(daten[0])[1:-1]}" 
        daten = f" | {str(daten[0])[1:-1]}" 
        pred= vowpal_workspace.predict(daten)
       
        prediction_result.append(pred)
    returned_df =pd.DataFrame({'predicted_actionID': prediction_result,  }, columns=['predicted_actionID',  ])
    return returned_df






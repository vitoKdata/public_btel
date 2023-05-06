import streamlit as st
import numpy as np

import vowpalwabbit as pyvw
import streamlit as st
import snowflake.connector


import pandas as pd
import numpy as np
import uuid
# Import Snowpark functions
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import col, lag
from snowflake.snowpark.window import Window
import snowflake.connector
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



account = st.secrets["account"]
username = st.secrets["user"] 
password = st.secrets["password"] 
database = st.secrets["database"] 

connection_parameters = {
   "account": account,
   "user": username,
   "password": password,
   "warehouse": "compute_wh",
   "role": "accountadmin",
   "database": database,
   "schema": "DATA"
}

def create_snowflake_connection():
    conn = snowflake.connector.connect(
        account=account,
        password=password,
        user=username,
        warehouse="compute_wh",
        database=database,
        schema="DATA"
    )

    return conn

def create_session_object():
    """Create Snowpark session object"""
    connection_parameters = st.secrets
    session = Session.builder.configs(connection_parameters).create()
    return session






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
        context_dict = context.to_dict(orient='records')
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
        context_dict = context.to_dict(orient='records')
        bandit_loop.reset_index(inplace=True, drop=True)

        daten =[
        vowpal_wabit_context_string(list(my_dict.items())[0]),
            

        ]
        
        #daten = f"{bandit_loop.Action.iloc[0 ]}:{reward.iloc[0]}:{prob_action.iloc[0 ]} | {str(daten[0])[1:-1]}" 
        daten = f" | {str(daten[0])[1:-1]}" 
        pred= vowpal_workspace.predict(daten)
       
        prediction_result.append(pred)
    returned_df =pd.DataFrame({'predicted_actionID': prediction_result,  }, columns=['predicted_actionID',  ])
    return returned_df









# Display the image with a width of 300 pixels
st.image("./images/elbandito_logo.PNG",  width=300)














st.markdown("""

# El Bandito Restaurant Recommendation

"""
)
# progress_bar = st.sidebar.progress(0)
# status_text = st.sidebar.empty()

# last_rows = np.random.randn(1,1)
# chart = st.line_chart(last_rows)

# for i in range(1,101):
#     new_rows = last_rows[-1,:] + np.random.randn(5,1).cumsum(axis=0)
#     status_text.text("%i%% Complete"%i)
#     progress_bar.progress(i)
#     chart.add_rows(new_rows)
#     last_rows = new_rows
    
#     time.sleep(0.1)
    
# progress_bar.empty()

# st.button('Re-run')

import streamlit as st 
import pandas as pd
import os 
from datetime import datetime

st.markdown("""

Welcome to our restaurant recommendation application! To help you find the perfect dining spot, we need some information about your preferences. The more details you provide, the better we can match you with the best restaurants in Mexico. We will use your location to suggest nearby options, and we will ask about your favorite cuisines, dietary restrictions, price range, and ambiance preferences to narrow down our recommendations.

We will also ask for your previous restaurant experiences to learn more about your taste and expectations. This information will be used to tailor our suggestions to your liking, and improve our algorithm for future recommendations. Don't worry, your data is securely stored and will not be shared with any third party without your consent.

Thank you for trusting us to help you discover your next favorite restaurant in Mexico. Let's get started!

"""
)
# Create Session State object

# Get the session state



# Define a fo



session_state = st.session_state
if 'form_submitted' not in session_state:
    session_state.form_submitted = False	
    
side_bar = st.sidebar
side_bar.caption('Please fill in')
sub = False
sub_popout= False
with side_bar.form('myform'):





	VALIDATED_PARKING = st.selectbox('Features extended parking',('Yes','No'))

	MON_TUE_WED_THU_FRI = st.selectbox('Open during the week',('Yes',
		'No',))
	SAT = st.selectbox('Open on saturday',('Yes',
		'No',))
	SUN = st.selectbox('Open on sunday',('Yes',
		'No',))
	YES = st.selectbox('Family friendly',('Yes',
		'No',))
	BAKERY = st.selectbox('Offers morning meals',('Yes',
		'No',))
	BREAKFAST_BRUNCH = st.selectbox('Offers brunch',('Yes',
		'No',))
	BAR = st.selectbox('Has a bar',('Yes',
		'No',))
	BAR_PUB_BREWERY = st.selectbox('Has its own brewery',('Yes',
		'No',))
	BARBECUE = st.selectbox('has barbecue',('Yes',
		'No',))
	Food = st.selectbox('Food',('MEXICAN',
		'INTERNATIONAL',))
	VEGETARIAN = st.selectbox('Has vegetarian food',('Yes',
		'No',))
	
	submitted = st.form_submit_button('Submit')

	if submitted:
		sub_popout = True
		# session_state.form_submitted = True
		side_bar.success('Form Subimitted Sucessfully')
		details = {
		
		'VALIDATED_PARKING': VALIDATED_PARKING,
		'MON_TUE_WED_THU_FRI': MON_TUE_WED_THU_FRI,
		'SAT':SAT,
		'SUN': SUN,
        'YES': YES,
		'BAKERY':BAKERY,
		'BAR': BAR,
		'BAR_PUB_BREWERY':BAR_PUB_BREWERY,
		'BARBECUE':BARBECUE,
		'BREAKFAST_BRUNCH': BREAKFAST_BRUNCH,
		'Food': Food, 
		'VEGETARIAN':VEGETARIAN

		}
		df = pd.DataFrame(details.items(), columns=['Category', 'Value'])
		
		sub = True

		df = df.set_index('Category').T
		df['MEXICAN'] = np.where(df['Food'] == 'MEXICAN', 1, 0)
		df['INTERNATIONAL'] = np.where(df['Food'] == 'INTERNATIONAL', 1, 0)
		df = df.replace({'Yes': 1, 'No': 0})
		df['generatedID'] = [uuid.uuid4() for _ in range(len(df))]
		preddd = pyvw.Workspace(f"--cb 112 -i cb.snowflake_bandit")
		a= recommending_cb(df, preddd)
		session = create_session_object()
		snow_df_pce = (session.table("BANDIT.DATA.RATINGS")) 
		snow_df_pce.show()
		data = snow_df_pce.to_pandas()
		if "rating2" not in st.session_state:
			st.session_state.rating2 = None
		if "submitted2" not in st.session_state:
			st.session_state.submitted2 = False
		b=placeIDgeben(data, 'PLACEID', a['predicted_actionID'].iloc[0])
		if not st.session_state.submitted2:
			rating_options = ["1 - Poor", "2 - Fair", "3 - Good", "4 - Very Good", "5 - Excellent"]
			user_rating = st.radio("Select a rating:", rating_options)
			st.session_state.submitted2 = st.sidebar.button("Submit and restart", key="submit2")
		if st.session_state.submitted2:
			st.write("Thank you for your ratings!")
            
        





# ... (your code here)

# ... (your code here)

# ... (your code here)

# ... (your code here)

# Initialize session_state variables
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False

if 'rating' not in st.session_state:
    st.session_state.rating = 1

if 'rating_submitted' not in st.session_state:
    st.session_state.rating_submitted = False

if 'expander_expanded' not in st.session_state:
    st.session_state.expander_expanded = False

# ... (your code here)

with st.expander("Results", expanded=True):
    if not sub:
        st.write("Please submit first.")
    else:
        df = get_place_data(b) 
        st.write(f"We recommend eating at {df['Name'].iloc[0]}") 
        st.write(f"  {df['Advertisment'].iloc[0]}")
        st.image(f"{df['Image_url'].iloc[0]}", caption="Looks delicious, doesn't it?")
        st.session_state.expander_expanded = True










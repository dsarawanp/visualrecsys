import streamlit as st
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import snowflake.connector
import random

from smart_open import open


from azure.storage.blob import BlobServiceClient

connect_str = 'DefaultEndpointsProtocol=https;AccountName=visrecstorage;AccountKey=q3Wvmg9bF4oPqZYdXV6PJ2+XPDfD3z4FckngdyHGMyCGE5zHMgqKPVNVk3AxGdjERc28EHGBVEE2+AStDSPpVw==;EndpointSuffix=core.windows.net'
transport_params = {
    'client': BlobServiceClient.from_connection_string(connect_str),
}

st.title('Movie Poster Recommender System')


#Create a file method 
def file_name(uploaded_file):
    return "azure://uploads/"+ str(uploaded_file)+".jpg"


#Create a save file method 
def save_uploaded_file(data, uploaded_file):
    try:
        with open(file_name(uploaded_file),'wb', transport_params=transport_params) as f:
            f.write(data)
        return 1
    except:
        return 0

# Create a function to extract the feature of the image using model
def feature_extraction(img_path,model):
    img1 = open(img_path, 'rb', transport_params=transport_params)
    img = Image.open(img1).resize((224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# Create a function to recommend the images based on the features extract by the model.
def recommend(features,genre):
  # provide the path for the feature extraction file 
    feature_extraction_file='azure://extraction/'+genre+'_imageFeaturesEmbeddings.pkl'        
    feature_list = np.array(pickle.load(open(feature_extraction_file,'rb', transport_params=transport_params)))
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


def predict_movies(movie_id):
    # feature extract
        features = feature_extraction(file_name(movie_id),model)
        sql = "SELECT movie_title, genres FROM MOVIE_DATA where movie_id = "+str(movie_id)
        print(sql)
        cursor.execute(sql)
        txt=cursor.fetchall()
        print(txt[0][1])
        genres = txt[0][1].split('-')
        st.write(txt[0][0])
        st.write(txt[0][1])
        for genre in genres:
            # recommendention
            indices = recommend(features,genre)
            where_in = ','.join(['%s'] * len(indices[0]))
            # provide the path for the feature extraction file 
            feature_filenumber_file='azure://extraction/'+genre+'_imageFeaturesFileNumber.pkl'
            filenames1 = pickle.load(open(feature_filenumber_file,'rb', transport_params=transport_params))
            sql = "SELECT movie_id,movie_title,poster_image FROM movie_data where movie_id in (%s)" % (where_in)
            sql = sql+ "and genres like (%s) and movie_id not in (%s)"
            indices_list=[]
            length = len(indices[0])
            for i in range(0,length):
                indices_list.append(filenames1[indices[0][i]])
            tuple_list = tuple(indices_list) + ("%"+genre+"%", movie_id,)
            cursor.execute(sql,tuple_list)
            recomended_results = cursor.fetchall()
            recomended_result=[]
            for i in range (0,len(recomended_results)):
                for j in range(length):
                    if indices_list[j]==recomended_results[i][0]:
                        recomended_result.append(recomended_results[i])
                if len(recomended_result)==6:
                    break
            st.header(genre)
            col1,col2,col3,col4,col5 = st.columns(5)
            
            with col1:
                if save_uploaded_file(recomended_result[0][2],recomended_result[0][0]):
                # display the file
                    display_image0 = open(file_name(recomended_result[0][0]), 'rb', transport_params=transport_params)
                    display_image = Image.open(display_image0)
                    st.image(display_image)
                    st.write(recomended_result[0][1])
            with col2:
                if save_uploaded_file(recomended_result[1][2],recomended_result[1][0]):
                # display the file
                    display_image1 = open(file_name(recomended_result[1][0]), 'rb', transport_params=transport_params)
                    display_image = Image.open(display_image1)
                    st.image(display_image)
                    st.write(recomended_result[1][1])
            with col3:
                if save_uploaded_file(recomended_result[2][2],recomended_result[2][0]):
                # display the file
                    display_image2 = open(file_name(recomended_result[2][0]), 'rb', transport_params=transport_params)
                    display_image = Image.open(display_image2)
                    st.image(display_image)
                    st.write(recomended_result[2][1])
            with col4:
                if save_uploaded_file(recomended_result[3][2],recomended_result[3][0]):
                # display the file
                    display_image3 = open(file_name(recomended_result[3][0]), 'rb', transport_params=transport_params)
                    display_image = Image.open(display_image3)
                    st.image(display_image)
                    st.write(recomended_result[3][1]) 
            with col5:
                if save_uploaded_file(recomended_result[4][2],recomended_result[4][0]):
                # display the file
                    display_image4 = open(file_name(recomended_result[4][0]), 'rb', transport_params=transport_params)
                    display_image = Image.open(display_image4)
                    st.image(display_image)
                    st.write(recomended_result[4][1])

if "refreshclick" not in st.session_state:
    st.session_state.refreshclick=False
    if "movie_id" not in st.session_state:
        randomlist=[]
        for i in range(0,30):
            n = random.randint(1,122)
            randomlist.append(n)
        st.session_state.movie_id=randomlist

        
try:
    ctx = snowflake.connector.connect(user='SARAWANPERNETI193',password='@Temp2023',account='hb81971.ca-central-1.aws',role="ACCOUNTADMIN",warehouse="COMPUTE_WH",database="MOVIES_DB",schema="MOVIES_TABLES")
    cursor = ctx.cursor()
    print ("randon number",tuple(st.session_state.movie_id))
    where_in = ','.join(['%s'] * len(st.session_state['movie_id']))
    sql = "SELECT movie_id,movie_title,poster_image FROM movie_data where movie_id in (%s) " % (where_in)
    sql = sql+ "and genres not like (%s) and genres not like (%s) and genres not like (%s)"
    print(sql)
    tuple_list = tuple(st.session_state.movie_id) + ("%TV Movie%","%Romance%","%Drama%")
    print(len(tuple_list))
    cursor.execute(sql, tuple_list)
    myresult = cursor.fetchall()
    col1,col2,col3,col4,col5 = st.columns(5)
    st.session_state.refreshclick = False
    if(len(myresult)!=0):
        with col1:
            if save_uploaded_file(myresult[0][2],myresult[0][0]):
            # display the file
                display_image0 = open(file_name(myresult[0][0]), 'rb', transport_params=transport_params)
                display_image = Image.open(display_image0)
                clicked_0 = st.button(myresult[0][1] , st.image(display_image))
        with col2:
            if save_uploaded_file(myresult[1][2],myresult[1][0]):
            # display the file
                display_image1 = open(file_name(myresult[1][0]), 'rb', transport_params=transport_params)
                display_image = Image.open(display_image1)
                clicked_1 = st.button(myresult[1][1] , st.image(display_image))

        with col3:
            if save_uploaded_file(myresult[2][2],myresult[2][0]):
            # display the file
                display_image2 = open(file_name(myresult[2][0]), 'rb', transport_params=transport_params)
                display_image = Image.open(display_image2)
                clicked_2 = st.button(myresult[2][1] , st.image(display_image))
        with col4:
            if save_uploaded_file(myresult[3][2],myresult[3][0]):
            # display the file
                display_image3 = open(file_name(myresult[3][0]), 'rb', transport_params=transport_params)
                display_image = Image.open(display_image3)
                clicked_3 = st.button(myresult[3][1] , st.image(display_image))

        with col5:
            if save_uploaded_file(myresult[4][2],myresult[4][0]):
            # display the file
                display_image4 = open(file_name(myresult[4][0]), 'rb', transport_params=transport_params)
                display_image = Image.open(display_image4)
                clicked_4 = st.button(myresult[4][1] , st.image(display_image))

        if(clicked_0):
            predict_movies(myresult[0][0])
        if(clicked_1):
            predict_movies(myresult[1][0])
        if(clicked_2):
            predict_movies(myresult[2][0])
        if(clicked_3):
            predict_movies(myresult[3][0])
        if(clicked_4):
            predict_movies(myresult[4][0])
finally:
    cursor.close()
ctx.close()

clicked = st.button("Refresh")
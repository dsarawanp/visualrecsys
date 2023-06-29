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
import mysql.connector as mysql
from mysql.connector import Error
import random

st.title('Movie Poster Recommender System')


#Create a file method 
def file_name(uploaded_file):
    return "uploads/"+ str(uploaded_file)+".jpg"


#Create a save file method 
def save_uploaded_file(data, uploaded_file):
    try:
        with open(file_name(uploaded_file),'wb') as f:
            f.write(data)
        return 1
    except:
        return 0

# Create a function to extract the feature of the image using model
def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# Create a function to recommend the images based on the features extract by the model.
def recommend(features,genre):
  # provide the path for the feature extraction file 
    feature_extraction_file='Extraction\\'+genre+'_imageFeaturesEmbeddings.pkl'        
    feature_list = np.array(pickle.load(open(feature_extraction_file,'rb')))
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
        sql = "SELECT movie_title, genres FROM movies.movie_data where movie_id = "+str(movie_id)
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
            feature_filenumber_file='Extraction\\'+genre+'_imageFeaturesFileNumber.pkl'
            filenames1 = pickle.load(open(feature_filenumber_file,'rb'))
            sql = "SELECT movie_id,movie_title,poster_image FROM movies.movie_data where movie_id in (%s)" % (where_in)
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
                    display_image = Image.open(file_name(recomended_result[0][0]))
                    st.image(display_image)
                    st.write(recomended_result[0][1])
            with col2:
                if save_uploaded_file(recomended_result[1][2],recomended_result[1][0]):
                # display the file
                    display_image = Image.open(file_name(recomended_result[1][0]))
                    st.image(display_image)
                    st.write(recomended_result[1][1])
            with col3:
                if save_uploaded_file(recomended_result[2][2],recomended_result[2][0]):
                # display the file
                    display_image = Image.open(file_name(recomended_result[2][0]))
                    st.image(display_image)
                    st.write(recomended_result[2][1])
            with col4:
                if save_uploaded_file(recomended_result[3][2],recomended_result[3][0]):
                # display the file
                    display_image = Image.open(file_name(recomended_result[3][0]))
                    st.image(display_image)
                    st.write(recomended_result[3][1]) 
            with col5:
                if save_uploaded_file(recomended_result[4][2],recomended_result[4][0]):
                # display the file
                    display_image = Image.open(file_name(recomended_result[4][0]))
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
    conn = mysql.connect(host='localhost', database='movies', user='root', password='@Temp2023')
    if conn.is_connected():
        cursor = conn.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)
        print ("randon number",tuple(st.session_state.movie_id))
        where_in = ','.join(['%s'] * len(st.session_state['movie_id']))
        sql = "SELECT movie_id,movie_title,poster_image FROM movies.movie_data where movie_id in (%s)" % (where_in)
        sql = sql+ "and genres not like (%s) and genres not like (%s) and genres not like (%s) "
        print(sql)
        tuple_list = tuple(st.session_state.movie_id) + ("%TV Movie%","%Romance%","%Drama%",)
        print(len(tuple_list))
        cursor.execute(sql, tuple_list)
        myresult = cursor.fetchall()
        col1,col2,col3,col4,col5 = st.columns(5)
        st.session_state.refreshclick = False
        if(len(myresult)!=0):
            with col1:
                if save_uploaded_file(myresult[0][2],myresult[0][0]):
                # display the file
                    display_image = Image.open(file_name(myresult[0][0]))
                    clicked_0 = st.button(myresult[0][1] , st.image(display_image))
            with col2:
                if save_uploaded_file(myresult[1][2],myresult[1][0]):
                # display the file
                    display_image = Image.open(file_name(myresult[1][0]))
                    clicked_1 = st.button(myresult[1][1] , st.image(display_image))
                    
            with col3:
                if save_uploaded_file(myresult[2][2],myresult[2][0]):
                # display the file
                    display_image = Image.open(file_name(myresult[2][0]))
                    clicked_2 = st.button(myresult[2][1] , st.image(display_image))
            with col4:
                if save_uploaded_file(myresult[3][2],myresult[3][0]):
                # display the file
                    display_image = Image.open(file_name(myresult[3][0]))
                    clicked_3 = st.button(myresult[3][1] , st.image(display_image))
                    
            with col5:
                if save_uploaded_file(myresult[4][2],myresult[4][0]):
                # display the file
                    display_image = Image.open(file_name(myresult[4][0]))
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
                        
    else:
        st.header("Data Base Connection issue")
except Error as e:
   print(e)
clicked = st.button("Refresh")
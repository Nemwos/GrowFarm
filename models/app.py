import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
import torch
import warnings
import requests
from streamlit import config
from PIL import Image
from torchvision import models, transforms
from fertilizer import fertilizer_dic
from utils.model import ResNet9
from utils.disease import disease_dic
from streamlit_lottie import st_lottie



    
#------------------------- READING FILES and CLASSES -------------------------#

fertilizer_df = pd.read_csv('models/data/fertilizer.csv')

disease_classes =['Apple___Apple_scab',
               'Apple___Black_rot',
               'Apple___Cedar_apple_rust',
               'Apple___healthy',
               'Blueberry___healthy',
               'Cherry_(including_sour)___Powdery_mildew',
               'Cherry_(including_sour)___healthy',
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)___Common_rust_',
               'Corn_(maize)___Northern_Leaf_Blight',
               'Corn_(maize)___healthy',
               'Grape___Black_rot',
               'Grape___Esca_(Black_Measles)',
               'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
               'Grape___healthy',
               'Orange___Haunglongbing_(Citrus_greening)',
               'Peach___Bacterial_spot',
               'Peach___healthy',
               'Pepper,_bell___Bacterial_spot',
               'Pepper,_bell___healthy',
               'Potato___Early_blight',
               'Potato___Late_blight',
               'Potato___healthy',
               'Raspberry___healthy',
               'Soybean___healthy',
               'Squash___Powdery_mildew',
               'Strawberry___Leaf_scorch',
               'Strawberry___healthy',
               'Tomato___Bacterial_spot',
               'Tomato___Early_blight',
               'Tomato___Late_blight',
               'Tomato___Leaf_Mold',
               'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite'
               'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
               'Tomato___Tomato_mosaic_virus',
               'Tomato___healthy']


disease_model_path = 'models/plantdisease.pth'
disease_model = torch.load(disease_model_path, map_location=torch.device('cpu'))
print(disease_model.eval())


#------------------------- CREATING ARRAY FOR CITIES -------------------------#

state_arr = ["Andaman & Nicobar", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chandigarh", "Chhattisgarh", "Dadra & Nagar Haveli", "Daman & Diu", "Delhi", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jammu & Kashmir", "Jharkhand", "Karnataka", "Kerala", "Lakshadweep", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Orissa", "Pondicherry", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Tripura", "Uttar Pradesh", "Uttaranchal", "West Bengal"]



#------------------------- DEFINING LOTTIE FILES -------------------------#

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()



#------------------------- LOADING ASSETS -------------------------#

lottie_earth = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_jggbnfb8.json")
lottie_garden = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_kneqbtiw.json")
lottie_sow = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_hhoese47.json")
lottie_plant = load_lottieurl("https://assets2.lottiefiles.com/private_files/lf30_hcrwoq8z.json")



#------------------------- DEFINING FERTILIZER -------------------------#

def predict_fertilizer(crop_name,N,P,K):
    global fertilizer_df
    
    nr = fertilizer_df[fertilizer_df['Crop'] == crop_name]['N'].iloc[0]
    pr = fertilizer_df[fertilizer_df['Crop'] == crop_name]['P'].iloc[0]
    kr = fertilizer_df[fertilizer_df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"
    resp = str(fertilizer_dic[key])
    return resp



#------------------------- LOADING MODELS -------------------------#

def load_model(model_name):
    path = model_name
    file = open(path,'rb')
    model = pickle.load(file)
    file.close()
    return model



#------------------------- IMAGE PROCESSING -------------------------#

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction



#------------------------- CUSTOM FUNCTIONS -------------------------#


def weather_fetch(city_name):
    api_key = "<weather_api_key>"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x.get("main", {})
        temperature = round((y.get("temp", 0) - 273.15), 2)
        humidity = y.get("humidity", 0)
        return temperature, humidity
    else:
        return None

    

#------------------------- HOME PAGE -------------------------#

def home():
    st.markdown("<h1 style='text-align:center'>Welcome to GrowFarm</h1>", unsafe_allow_html=True)
        
    image = Image.open('images/first.png')
    image = image.resize((800,200))
    st.image(image, use_column_width=True)
    
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st. header("What we Focus on")
            st.write(
                """
                """)
            st.write(
                """
                On this Website you will find ways and solutions to the problems related to farming and harvesting your crops:
                
                Our primary focus is on developing a system for predicting crops based on soil data, predicting fertilizers based on soil and crop details, and identifying plant diseases. Our system's goal is to assist farmers since it is difficult to cultivate crops when you don't know the weather. Furthermore, it involves a number of soil and crop-related elements.
                These outcomes are determined by the soil condition.
                """
                )

        with right_column:
            st_lottie(lottie_earth, height =450, key="earth")
            

        with st.container():
            st.write(
                """

                """)
            st.header("Services we Provide")
            st.write(
                """

                """)
            col1, col2, col3 = st.columns(3)

            with col1:
                st_lottie(lottie_garden, height =200, key="garden")
                st.markdown("<div style='text-align:center'><h3><b>Crop Recommender</b></h3></div>", unsafe_allow_html=True)
                st.write(
                """
                """)
                st.write("<div style='text-align:center'><h6>A Recommender that helps to know which crop is best to grow based on soil attributes and rain data provided</h6></div>", unsafe_allow_html=True)

            with col2:
                st_lottie(lottie_sow, height =200, key="sow")
                st.markdown("<div style='text-align:center'><h3><b>Fertilizer Prediction</b></h3></div>", unsafe_allow_html=True)
                st.write(
                """
                """)
                st.write("<div style='text-align:center'><h6>This gives the prediction about what fertilizers is requiresd for your soil to grow based on provided data</h6></div>", unsafe_allow_html=True)                              
                
            with col3:
                st_lottie(lottie_plant, height =200, key="plant")
                st.markdown("<div style='text-align:center'><h3><b>Plant Disease Prediction</b></h3></div>", unsafe_allow_html=True)
                st.write(
                """
                """)
                st.write("<div style='text-align:center'><h6>This gives the prediction about the disease that your plant got based on the image provided</h6></div>", unsafe_allow_html=True)


#------------------------- CROP RECOMMENDER PAGE -------------------------#

def print_state():
    state_id = st.selectbox("Select State", ["Select State"] + state_arr)
    return state_id

def Crop_Recommender():
    
    # page title
    st.title("Crop Recommender")

    # getting the input data from the user
    col1, col2 = st.columns(2)


    with col1:
        with st.expander(" ℹ️ Information", expanded=True):
            st.write(""" Recommendation about the type of crops to be cultivated which is best suited for the respective conditions.
                """)
            '''
            ## How does it work ❓ 
            Complete all the parameters and the machine learning model will predict the most suitable crops to grow in a particular farm based on various parameters
        '''

    
    with col2:
        st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        N = st.number_input("Nitrogen", 1, 10000)
        P = st.number_input("Phosphorus", 1, 10000)
        K = st.number_input("Potassium", 1, 10000)
        temp = st.number_input("Temperature", 0.0, 100000.0)
        humidity = st.number_input("Humidity in %", 0.0, 100000.0)
        ph = st.number_input("Ph", 0.0, 100000.0)
        rainfall = st.number_input("Rainfall in mm", 0.0, 100000.0)
        
        state = st.selectbox("Select State", ["Select State"] + state_arr)

        
        
        # Convert data to pandas df
        single_pred = {
            "N": N,
            "P": P,
            "K": K,
            "temperature": temp,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }

        pred_df = pd.DataFrame([single_pred])
        if st.button('Predict'):
            loaded_model = load_model('models/RandomForest.pkl')
            prediction = loaded_model.predict(pred_df)
            col1.write('''
        		    ## Results 🔍 
        		    ''')
            col1.success(f"{prediction.item().title()} are recommended by the A.I for your farm.")



#------------------------- FERTILIZER PREDICTION PAGE -------------------------#

def Fertilizers_Prediction():
    # page title
    st.title('Fertilizers Prediction')

    col1, col2 = st.columns(2)

    with col1:
        with st.expander(" ℹ️ Information", expanded=True):
            st.write("""  Recommendation about the type of fertilizer best suited for the particular soil and the recommended crop
            """)

        '''
        ## How does it work ❓ 
        Complete all the parameters and the machine learning model will predict the most suitable crops to grow in a particular farm based on various parameters
        '''

    with col2:
        crops = fertilizer_df['Crop'].unique().tolist()
        st.subheader(" Get informed advice on fertilizer based on soil 👨‍🌾")
        crop_name = st.selectbox("Crop Name",crops)
        N = st.number_input("Nitrogen", 1, 10000)
        P = st.number_input("Phosphorus", 1, 10000)
        K = st.number_input("Potassium", 1, 10000)


        feature_list = [N, P, K]
        single_pred = np.array(feature_list).reshape(1, -1)
        st.write("""

        """)
        st.empty()

    if st.button("Predict"):
        st.write("<div style='text-align: center;'><h3><b> Prediction 🔍 </b></h4>", unsafe_allow_html=True)
        prediction = predict_fertilizer(crop_name, N, P, K)
        st.markdown("<div style='text-align: center;'><p>" + prediction + "</p></div>", unsafe_allow_html=True)            

# for moving data into GPU (if available)
def get_default_device():
    return torch.device("cpu")

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def predict_image(img, model):
    """Converts image to array and return the predicted class
        with highest probability"""
    # Convert to a batch of 1
    try:
        xb = to_device(img.unsqueeze(0), device)
        # Get predictions from model
        yb = model(xb)
        # Pick index with highest probability
        _, preds  = torch.max(yb, dim=1)
        # Retrieve the class label
        labels = ['Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy',
            'image___not_found'
        ]

        if labels[preds[0].item()] == 'image___not_found':
            return 'image not found'

        return labels[preds[0].item()]

    except:
       return 'image not found'

device = get_default_device()

transform = transforms.Compose([
    transforms.ToTensor()
])



#------------------------- PLANT DISEASE PREDICTION PAGE -------------------------#

def Plant_Disease_Prediction():
    st.title("Plant Disease Prediction")

    file_up = st.file_uploader("Upload a Photo",type=['png','jpg','jpeg'])
    if file_up is None:    
        return
    print(file_up)
    image = Image.open(file_up)
    
    if st.button("Predict"):
        slot = st.empty()

           # img = file.read()
        slot.text('Running inference....')
        st.image(image, caption="Input Image", width = 300)
        image = transform(image)
        loaded_model = torch.load('models/plantdisease.pth')
        # prediction = predict_image(image,loaded_model)
        try:
            prediction = predict_image(image, loaded_model)
        except ValueError:
            st.error('Image not found in model. Please upload a valid image.')
            return
        if prediction == 'image not found':
            slot.empty()
            st.error('The uploaded image was not found in the model. Please upload a different image.')
            return

        slot.empty()
        st.write(
            """
            """)
        prediction = disease_dic[prediction]
        st.markdown("<div style='overflow:hidden;'>Prediction: {}".format(prediction), unsafe_allow_html=True)           

            

#------------------------- MAIN FUNCTION -------------------------#
            
def main():
    st.set_page_config(page_title="Recommender System", page_icon = ":herb:", layout="wide")

    # sidebar for navigation
    from streamlit_option_menu import option_menu
   
    selected = option_menu(None, ["HOME","Crop Recommend","Fertilizer Prediction","Plant Disease Prediction"], 
        icons=['house', 'search', "list-task","cloud-upload"], 
        menu_icon="cast", default_index=0, orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "00FFAB"},
            "icon": {"color": "white", "font-size": "25px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"20px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "D7E9B9"},
        }
    )

    if (selected == 'HOME'):
        home()
    if (selected == 'Crop Recommend'):
        Crop_Recommender()
    if (selected == 'Fertilizer Prediction'):
        Fertilizers_Prediction()
    if (selected == "Plant Disease Prediction"):
        Plant_Disease_Prediction()

if __name__=="__main__":
    main()

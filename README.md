# **GrowFarm – Crop, Fertilizer and Plant Disease Prediction using Machine Learning**
A simple ML and DL based website made with Streamlit which recommends the best crop to grow, fertilizers to use and the diseases caught by your crops.

# **DISCLAIMER ⚠️**
This is a proof-of-concept (POC) project. The developer provides no assurance for the data used here. As a result, don't use it to make farming judgments. If you do this, the designer is not liable for anything. This project, on the other hand, illustrates the idea of how we can apply ML/DL in precision farming if developed on a large scale and with real and confirmed data.


# **Objective and scope of the project**
● The project aims to make a recommendation system of crops to develop a model capable of predicting crop sustainability in each state based on soil type and meteorological circumstances.
● It gives recommendations for the best crops in the region so that farmers can minimize their losses.
● The project also recommends which fertilizer to use, and how to care for plants and crops.
● Assists individuals in identifying crops and plants they are unfamiliar with, as well as detecting crop disease.
● It acts as a one stop solution for an individual to gather all basic information on plants and crops and their uses, benefits in one place.


# **Built with 🛠️**
Python
Machine Learning
Deep Learning
NumPy
Matplotlib
Pandas
Scikit Learn
Seaborn
PyTorch
Tensorflow
Streamlit


# **How to use 💻**
● Crop Recommendation System ==> Enter your soil, state, and city's nutrient values. It should be noted that the N-P-K (Nitrogen-Phosphorous-Pottasium) values should be entered in the ratio. More information can be found on this page. Make sure to enter largely common city names when entering the city name. Remote cities/towns may not be available in the Weather API, which fetches humidity and temperature data.

● Fertilizer recommendation system ==> Enter your soil's nutrient content and the crop you want to plant. The algorithm will determine which nutrients are in excess or deficient in the soil. As a result, it will make recommendations for fertilizer purchases.

● Disease Detection System ==> Upload a photo of a leaf from your plant. The algorithm will determine the crop kind and whether it is infected or healthy. If it is ill, it will tell you the reason of the illness and how to prevent/cure it. Please keep in mind that it currently only supports the crops listed below.

# ------------------------------ Streamlit MyWebsite Project 02 ------------------------------
#Project_Date : 04.08.2022
#Project Purpose : Make a website with streamlit dictionary
#Project Process :
    #0 -  Install and Import Packages (streamlit, Pillow, streamlit_lottie)
    #1 - Set Page Configrations title, icon, layout
    #2 - Use Local Css to config. Website (create style directory)
    #3 - Load Assets (lottie, images) (Create images directory and put the images here)
    # 4 - Header Section01
    # 5- What I do Section02
    # 6 - Projects
        # 6.1 - Project01 Section03
        # 6.2 - Project02 Section04
    # 7 - DataFrame Read
    # 8 - Slider, Selection Box, Text Input and ML Model
        # 8.1 - Read Taxi DataFrama
        # 8.2 - Create slider, selectbox, text_input
        # 8.3 - ML Model RandomForestRegressor
        # 8.4 - Errors and R-Squared
    # 9 - Contact Form (Send Mail to You)
# ------------------------------ ------------------------------ ------------------------------
import time
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore')

# Sleep function 
def sleep(x):
    time.sleep(x)

# Wait for a certain measure of time before throwing an exception
def wait(x):
    driver.implicitly_wait(x)

# Click Function
def click_bann_byID(ID):
    actions = ActionChains(driver)
    akzeptieren = driver.find_element(By.ID, ID)
    actions.click(akzeptieren).perform()
    wait(10)
    sleep(0.5)

# Find Elements Function
def find_elements_HPCO(H,P,C,O):
    if website_name == 'jobware':
        header = driver.find_elements(By.TAG_NAME, H)
    else:
        header = driver.find_elements(By.CLASS_NAME, H)
    publish = driver.find_elements(By.CLASS_NAME, P)
    company = driver.find_elements(By.CLASS_NAME, C)
    ort = driver.find_elements(By.CLASS_NAME, O) 

    list_header = [title.text for title in header]
    list_publish = [pub.text for pub in publish]
    list_company = [comp.text for comp in company]
    list_ort = [o.text for o in ort]
    return list_header, list_publish, list_company, list_ort

# Scroll Down Function
def scroll_down(x):
    n=0
    while n < x:
        n+=1
        actions.key_down(Keys.PAGE_DOWN).perform()
        sleep(1.5)
        actions.key_down(Keys.PAGE_DOWN).perform()
        sleep(1.5)
        actions.key_down(Keys.PAGE_DOWN).perform()
        sleep(1.5)
        actions.key_down(Keys.PAGE_UP).perform()
        sleep(0.10)
        actions.key_down(Keys.PAGE_DOWN).perform()
        wait(10)
        sleep(2.5)




        
        
        
        




import pandas as pd
import numpy as np
import seaborn as sns
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image


# 8.20 - Not to Run always Data Frame , It saves Data and not reload if DataFrame not changed
@st.cache
def get_data(filename):
    df = pd.read_csv(filename)
    return df

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# 1 - Set Page Configrations
st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")
# 1.1 - Set Background Color
st.markdown(
    '''
    <style>
     .main {
     background-color: #F5F5F5;
     }
    </style>
    ''',
    unsafe_allow_html=True)

# 2 - Use Local Css to config. Website

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}<style>", unsafe_allow_html=True)

local_css("style/style.css")


# 3 - Load Assets (lottie, images)
lottie_coding =load_lottieurl('https://assets10.lottiefiles.com/packages/lf20_0yfsb3a1.json')
img01_contact_form = Image.open('images/coding01.jpg')
img02_contact_form = Image.open(('images/coding02.jpg'))


# 4 - Headers Section01
st.subheader("Hi, I am Abdullah Cay :wave:")
st.title("A Data Analyst From Germany")
st.write("I am passionate about finding ways to use Python ")
st.write("[Learn More >](https://github.com/abdullahcayde)")


# 5- What I do Section02
st.write('---')
with st.container():
    left_column, right_column = st.columns(2)

    with left_column:
        st.header('What I do ?')
        st.write('##')
        st.write(
            '''
            I am learning everyday Coding with Python : 
            - .
            - .
            - .
            -  ......
            ''')
        st.write('[Learn more>](https://github.com/abdullahcayde)')

        #with right_column:
            #st_lottie(lottie_coding, height=300, key='coding')

# 6 - Projects
# 6.1 - Project01 - Pandas Visualization - Section03
with st.container():
    st.write('---')
    st.header('Pandas Visualization')
    st.write('##')
    image_column , text_column = st.columns((1,2))

    with image_column:
        st.image(img01_contact_form)
    with text_column:
        st.subheader('Learn How to visualize your Data with Pandas')
        st.write(
            '''
            Learn how to use Lottie Files in Streamlit !!!!!
        ''')
        st.markdown("[Github Link ...](https://github.com/abdullahcayde/Trainings/blob/main/Pandas_Visualization/Pandas_Visualization_05.ipynb)")

# 6.2 - Project02 - Statistics for Machine Learning - Section04
with st.container():
    st.write('---')
    st.header('Statistics for Machine Learning')
    st.write('##')
    image_column, text_column = st.columns((1, 2))

    with image_column:
        st.image(img02_contact_form)
    with text_column:
        st.subheader('Basic Hypothesis Testing, A/B Testing, Varieance Testing ... ')
        st.write(
            '''
            Learn how to use Lottie Files in Streamlit !!!!!
        ''')
        st.markdown("[Github Link ...](https://github.com/abdullahcayde/Trainings/tree/main/istatistik)")

# 7 - DataFrame Read
path = 'data'
name = 'ebay09_jazz'
use_cols =[1,2,3,4,5,6]

with st.container():
    st.write('---')
    st.header('Data Frame - Pandas')

    df_jazz = pd.read_csv(f'{path}/{name}.csv', usecols= use_cols)
    st.subheader('Data Frame- Jazz Head')
    st.write(df_jazz.head())

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader('Data Frame- Jazz Describe')
        st.write(df_jazz.describe())
    with right_column:
        st.subheader('Year - Bar Chart')
        st.bar_chart(pd.DataFrame(df_jazz['Year'].value_counts()))
        st.markdown('* **first feature:** I want see which Model Year Distribution')


# 8 - Slider, Selection Box, Text Input and ML Model
with st.container():
    st.write('---')
    st.title('Job Searching Project')
    
    #jobs_website = st.selectbox('Which websites do you want to select ?', options= ['Stepstone', 'Jobware', 'Linkedin'], index= 0 )
    
    jobs_website = st.write('Which websites do you want to select ?')
    option_1 = st.checkbox('Stepstone')
    option_2 = st.checkbox('Jobware')
    option_3 = st.checkbox('Linkedin')
    
    jobs_searchWords = st.selectbox('Which job title do you want to select ?', options= ['Business Analyst', 'Data Scientist', 'Data Analyst'], index= 0 )
    
    
    st.write('Selected Job Title :',  jobs_searchWords)
    st.write('Selected Websites : ', )
    
    if option_1:
        print('---------------------- StepStone Job Searching Selenium Project ----------------------')
        start=datetime.now()  
        # Link Descriptions
        link_original_stepstone = 'https://www.stepstone.de/jobs/data-analyst/in-rietberg?radius=50&page=2'

        website_name = option_1
        job_name = jobs_searchWords
        ort_ = 'Rietberg'
        radius = 50
        page_number = 1

        #  1 - Create Driver
        Path = '/Users/macbook/Desktop/projects/Github_Repositories/Python_Projects/pythonProjectSelenium/data/chromedriver'
        driver = webdriver.Chrome(Path)

        #  2 - Go to Website
        job_link = job_name.replace(' ', '-').lower()
        ort_link = ort_.lower()
        link = f'https://www.stepstone.de/jobs/{job_link}/in-{ort_link}?radius={radius}&page={page_number}'

        driver.get(link)
        wait(10)
        sleep(2)

        #  3 - ActionChain Object created
        # 3.1 - Click Banned Accept
        ID = 'ccmgt_explicit_accept'
        click_bann_byID(ID)


        # 4 -  Take Infos from Page
        # 4.1 - Headers, Publish_Time ,Company, City
        H, P, C, O = 'resultlist-1uvdp0v', 'resultlist-w7zbt7', 'resultlist-1va1dj8', 'resultlist-suri3e'
        list_header, list_publish, list_company, list_ort = find_elements_HPCO(H,P,C,O)

        # 4.2 - Description and Page number of results
        description = driver.find_elements(By.CLASS_NAME, 'resultlist-1fp8oay')
        result = driver.find_elements(By.CLASS_NAME, 'resultlist-1jx3vjx')


        # 4.3 - Get Links
        header = driver.find_elements(By.CLASS_NAME, H)
        list_link = [link.get_attribute('href') for link in header]

        # 4.4 - Get Texts for each finding
        list_description = [des.text for des in description]
        print('Header',len(list_header), 'Publish',len(list_publish), 'Company',len(list_company[1:]), 'Ort',len(list_ort), 'Desc', len(list_description), 'Link',len(list_link))

        # 4.5 - Total Search Page Number
        list_result = [res.text for res in result]
        number_of_page = int(list_result[-2])
        print(f'Number of Jobs Pages = {number_of_page}')

        # 4.6 - DataFrame df
        d = dict(job_title=np.array(list_header), publish=np.array(list_publish), company=np.array(list_company[1:]), city=np.array(list_ort) , description=np.array(list_description), link=np.array(list_link))
        df = pd.DataFrame.from_dict(d, orient='index')
        df = df.T


        # 4.7 Repeat Process for every Web Page
        while  page_number < number_of_page:
            page_number+=2

            # 4.7.1 - Go to another page
            link = f'https://www.stepstone.de/jobs/{job_link}/in-{ort_link}?radius={radius}&page={page_number}'
            driver.get(link)
            wait(10)
            sleep(1.5)

            # 4.7.2 - Find the elements and get the Texts
            list_header, list_publish, list_company, list_ort = find_elements_HPCO(H,P,C,O) 
            description = driver.find_elements(By.CLASS_NAME, 'resultlist-1pq4x2u')
            list_description = [des.text for des in description]
            header = driver.find_elements(By.CLASS_NAME, H)
            list_link = [link.get_attribute('href') for link in header]

            # 4.7.3 - Create new page Dataframe
            d = dict(job_title=np.array(list_header), publish=np.array(list_publish), company=np.array(list_company[1:]), city=np.array(list_ort) , description=np.array(list_description), link=np.array(list_link))
            df2 = pd.DataFrame.from_dict(d, orient='index')
            df2 = df2.T

            # 4.7.4 - Concatenate the DataFrames
            df = pd.concat([df,df2], axis=0, ignore_index=True)
            print(f'Page Number : {page_number}, DataFrame Shape : {df2.shape}')


        # 5.1 - Save Data as csv 
        print(f'DataFrame End : {df.shape}')
        df['website'] = website_name
        time_ = datetime.today().strftime('%Y-%m-%d')
        df['date'] = time_
        job_name2 = job_name.replace(' ', '_')
        df['search_title'] = job_name2

        path = '/Users/macbook/Desktop/projects/Github_Repositories/Portfolio Projects/02 - Web_Scraping_Job_Search/data'
        job_name3 = job_name.replace(' ', '-')
        time_ = datetime.today().strftime('%Y-%m-%d')
        #df.to_csv(f'{path}/{job_name3}-{time_}.csv', index=False)

        # 6 - Quit
        end =datetime.now() 
        print('Code Runned No Problem')
        print(f'Time = {end - start}')
        sleep(2)
        driver.quit()
        
        # 7 - DataFrame Read

        with st.container():
            st.write('---')
            st.header('Data Frame - Pandas')

            st.subheader(f'Data Frame : {jobs_searchWords}')
            st.write(df.head())

            left_column, right_column = st.columns(2)
            with left_column:
                st.subheader(f'Data Frame : {jobs_searchWords}')
                st.write(df['company'].value_counts().head(10))
            with right_column:
                st.subheader('Job_Title - Bar Chart')
                st.bar_chart(pd.DataFrame(df['job_title'].value_counts().head()))
                #st.markdown('* **first feature:** I want see which Model Year Distribution')
    
    df['job_title'].value_counts().head()
    
    

# 9 - Contact Form (Send Mail to You)
with st.container():
    st.write('--')
    st.header('Get In Touch With Me!')
    st.write('##')

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    contact_form = '''
    <form action="https://formsubmit.co/abdullahcay26@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    '''
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()

        
        
        
        
        
        
        
        
      
'''
    set_col, disp_col = st.columns(2)

    # 8.1 - Read Taxi DataFrama
    set_col.subheader('Data Frame -TAXI Head')
    df = get_data('data/taxis.csv')
    set_col.write(df.head())

    # 8.2 - Create slider, selectbox, text_input
    #max_depth = set_col.slider('What should be the max_depth of the model ?', min_value=1, max_value=36, step=1)
    n_estimators = set_col.selectbox('What shoul be the n_estimators ?', options= [100, 200, 300, 'No Limit'], index= 0 )
    #input_feature = set_col.text_input('Select your Feature for the prediction of "Total :"', 'distance',placeholder='Feature')

    if input_feature == 'No Limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    set_col.text('Here is the list of features in my data :')
    set_col.table(df.columns)

    disp_col.subheader('Data Frame - TAXI Describe ')
    disp_col.write(df.describe().T)

    # 8.3 - ML Model RandomForestRegressor
    #if input_feature.dtype() != (float, int64
    regr = RandomForestRegressor(max_depth= max_depth, n_estimators= n_estimators)

    X = df[[input_feature]]
    y = df['total']

    regr.fit(X, y)
    prediction = regr.predict(X)

    # 8.4 - Errors and R-Squared
    disp_col.subheader('Mean Absolute Error is:')
    mae = disp_col.write(mean_absolute_error(y, prediction))


    disp_col.subheader('Mean Squared Error is:')
    mse = disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('R Squared Score of the Model is:')
    r_sqr = disp_col.write(r2_score(y, prediction))

'''







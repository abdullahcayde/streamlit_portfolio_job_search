import pandas as pd
import numpy as np
import seaborn as sns
import os
import openpyxl

import streamlit as st
import requests
from bs4 import BeautifulSoup
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


#url = 'https://www.ebay-kleinanzeigen.de/s-jobs/rietberg/data-analyst/k0c102l1363r50'

#url = 'https://www.jobware.de/jobsuche?jw_jobname=business%20analyst&jw_jobort=333**%20Rietberg&jw_ort_distance=50'
def get_data(url):
    
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:107.0) Gecko/20100101 Firefox/107.0'}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    return soup


def parse(soup):
    job_list = []
    results = soup.find_all('div', {'class' : 'aditem-main'})
    print(len(results))
    for item in results:
        job = {
            'location' : item.find('div', {'class' : 'aditem-main--top--left'}).text,
            'date' : item.find('div', {'class' : 'aditem-main--top--right'}).text,
            'title': item.find('a', {'class' : 'ellipsis'}).text,
            'company': item.find('a', {'class' : 'j-action j-dont-follow-vip'}).text
            
        }
        
        job_list.append(job)
    return job_list


def output(job_list):
    df = pd.DataFrame(job_list)
    for i in df.columns:
        df[i] = df[i].str.replace('\n' , '')
        df[i] = df[i].str.strip('')
    
    df.to_csv('ebay_job.csv', index=False)
    print('Save to CSV')
    return


#soup = get_data(url)
#job_list = parse(soup)
#output(job_list)


with st.container():
    st.write('---')
    st.title('Job Searching Project')
    
    #jobs_website = st.selectbox('Which websites do you want to select ?', options= ['Stepstone', 'Jobware', 'Linkedin'], index= 0 )
    
    
    jobs_searchWords = st.selectbox('Which job title do you want to select ?', options= ['Business Analyst', 'Data Scientist', 'Data Analyst'], index= 0 )
    
    jobs_website = st.write('Which websites do you want to select ?')
    option_1 = st.checkbox('Ebay')
    option_2 = st.checkbox('Jobware')
    option_3 = st.checkbox('Linkedin')
    
    if option_1:
        jobs_searchWords = jobs_searchWords.replace(' ', '-').lower()
        url = f'https://www.ebay-kleinanzeigen.de/s-jobs/rietberg/{jobs_searchWords}/k0c102l1363r50'

        def get_data(url):

            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:107.0) Gecko/20100101 Firefox/107.0'}
            r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.text, "html.parser")
            return soup


        def parse(soup):
            job_list = []
            results = soup.find_all('div', {'class' : 'aditem-main'})
            print(len(results))
            for item in results:
                job = {
                    'location' : item.find('div', {'class' : 'aditem-main--top--left'}).text,
                    'date' : item.find('div', {'class' : 'aditem-main--top--right'}).text,
                    'title': item.find('a', {'class' : 'ellipsis'}).text,
                    'company': item.find('a', {'class' : 'j-action j-dont-follow-vip'}).text

                }

                job_list.append(job)
            return job_list


        def output(job_list):
            df = pd.DataFrame(job_list)
            for i in df.columns:
                df[i] = df[i].str.replace('\n' , '')
                df[i] = df[i].str.strip('')

            #df.to_csv('ebay_job.csv', index=False)
            print('Save to CSV')
            return df


        soup = get_data(url)
        job_list = parse(soup)
        df = output(job_list)

        with st.container():
                st.write('---')
                st.header('Data Frame - Pandas')

                st.subheader(f'Data Frame : {jobs_searchWords}')
                st.dataframe(df.head())

                writer = pd.ExcelWriter(f'{jobs_searchWords}.xlsx')
                df.to_excel(writer, 'sheet1')
                writer.save()
                
                st.download_button('Download CSV',
                                  df.to_csv(),
                                  file_name = f'{jobs_searchWords}.csv',
                                  mime= 'text/csv')

                with open(f'{jobs_searchWords}.xlsx', "rb") as file:
                     btn = st.download_button(
                                label="Download Excel",
                                data=file,
                                file_name=f"{jobs_searchWords}.xlsx",
                                mime="text/csv")

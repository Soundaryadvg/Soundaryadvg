# Core Pkgs
import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np

# Utils
import os
import joblib

# Data Viz pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')



def main():
    """Mortality Prediction App"""
    st.title("Disease Mortality Prediction App")

    menu = ["Home","Login","SignUp"]
    submenu =["Plot","Pridiction","Metrics"]

    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader("Home")
        st.text("What is Hepatitis?")

    elif choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login"):
            if password == "1234":
                st.success("Welcome {}".format(username))
                st.success("Logged In as {}".format(username))

                activity = st.selectbox("Activity",submenu)
                if activity == "Plot":
                    st.subheader("Data Vis Plot")

                elif activity == "Prediction":
                    st.subheader("Prediction Analytics")

                   

            else:
                    st.warning("Incorrect Username/Password")
      
    elif choice == "SignUp":
        new_username = st.text_input("User name")
        new_password = st.text_input("Password",type='password')

        confirm_password = st.text_input("Confirm Password",type='password')
        if new_password == confirm_password:
            st.success("Password Confirmed")

        else:
            st.warning("Password not same")

        if st.button("Submit"):
           
            st.success("You have sucessfully create a new account")
            st.info("Login to Get Started")



if __name__== '__main__':
    main()    

            



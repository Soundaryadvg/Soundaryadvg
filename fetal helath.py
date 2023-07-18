import streamlit as st
import pandas as pd
import numpy as np
 


# Load EDA
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load Our Dataset
def load_data(data):
    df = pd.read_csv(data)
    return df


def main():
    st.title("Fetal Health")
    menu = ["Prediction","Training","Charts"]
    choice = st.sidebar.selectbox("Menu",menu)    
    if choice == "Prediction":
        st.subheader("Helath")

        # Methods :Context Fetal Health  Apporach (with)
        with st.form(key="Helath"):
            baselinevalue = st.text_input("baseline value")
            accelerations= st.text_input("accelerations")
            fetal_movement = st.text_input("fetal_movement")
            uterine_contractions = st.text_input("uterine_contractions")       
            prolongued_decelerations= st.text_input("prolongued_decelerations")
            abnormal_short_term_variability = st.text_input("abnormal_short_term_variability")
            percentage_of_time_with_abnormal_long_term_variability= st.text_input("percentage_of_time_with_abnormal_long_term_variability")
            histogram_mode= st.text_input("histogram_mode")
            histogram_mean = st.text_input("histogram_mean")
            histogram_median = st.text_input("histogram_median")
            histogram_variance = st.text_input("histogram_variance")
            submit_button = st.form_submit_button(label="Predicet")  

        if submit_button:
             import numpy as np
             td=np.array([[float(baselinevalue),float(accelerations),float(fetal_movement),float(uterine_contractions),float(prolongued_decelerations),float(abnormal_short_term_variability),float(percentage_of_time_with_abnormal_long_term_variability),float(histogram_mode),float(histogram_mean),float(histogram_median),float(histogram_variance)]])
             st.write(td)
             import joblib
             model=joblib.load('fetusmodel.pkl')
             pred= model.predict(td)
             st.write(pred[0])

    if choice == "Training":
        st.subheader("Training")
        df = load_data("fetal_health.csv")
        st.dataframe(df.head(100))
        X = df.drop(['light_decelerations', 'severe_decelerations','mean_value_of_short_term_variability', 'mean_value_of_long_term_variability','histogram_width','histogram_min','histogram_max', 'histogram_number_of_peaks','histogram_number_of_zeroes', 'histogram_tendency','fetal_health'],1)
        st.dataframe(X)
        y = df['fetal_health']
        st.dataframe(y)
        Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.20,random_state=46)
        clf=KNeighborsClassifier()
        import numpy as np
        ytrain=np.ravel(ytrain)
        clf.fit(Xtrain,ytrain)
        st.write("training complected")
        td=np.array([[120.0,0.0,0.0,0.0,0.0,0.62,0.3,0.12,0.68,0.1,0.2]])
        res=clf.predict(td)
        print(res)
        st.write(res[0])
        import joblib
        joblib.dump(clf,"fetusmodel.pkl")
          
 
    elif  choice == "Charts":
        # Data visualization
        import numpy as np
        import pandas as pd
        st.title("Bar chart")
        data=pd.DataFrame(np.random.randn(50,2),columns=["fetal_movement","percentage_of_time_with_abnormal_long_term_variability"])
        st.bar_chart(data)
        st.title("Line chart")
        st.line_chart(data)
        st.title("Area chart")
        st.area_chart(data)  
        

 
if __name__== '__main__':
    main()          
             
            



          


    
    
    
    
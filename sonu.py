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
    st.title("Maternal Health Risk")
    menu = ["Prediction","Training","Charts"]
    choice = st.sidebar.selectbox("Menu",menu)    
    if choice == "Prediction":
        st.subheader("Helath Risk")

        # Methods :Context Maternal Health Risk Apporach (with)
        with st.form(key="Helath"):        
            Age = st.text_input("Age")
            SystolicBP = st.text_input(" SystolicBP")
            DiastolicBP = st.text_input("DiastolicBP")
            BS = st.text_input("BS")
            HeartRate= st.text_input("HeartRate")
            Bodytemp= st.text_input("BodyTemp")
            
       
              
            submit_button = st.form_submit_button(label="Predicet")  


        if submit_button:
             import numpy as np
             td=np.array([[float(Age),float(SystolicBP),float(DiastolicBP),float(BS),float(HeartRate),float(Bodytemp)]])
             st.write(td)
             import joblib
             model=joblib.load('Maternalmodeal.pk1')
             pred= model.predict(td)
             st.write(pred[0])


    
    
    
    if choice == "Training":
        st.subheader("Training")
        df = load_data("Maternal Health Risk Data.csv")
        st.dataframe(df.head(100))
        # X Datasets
        X=df.iloc[:,[0,1,2,3,4,5]].values
        st.dataframe(X)
        # Y Datasets
        Y=df.iloc[:,[6]].values
        st.dataframe(Y)
        xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=50)
        clf=KNeighborsClassifier()
        import numpy as np
        import pandas as pd
        st.title("Bar chart")
        data=pd.DataFrame(np.random.randn(50,2),columns=["DiastolicBP","BP"])
        st.bar_chart(data)
        st.title("Line chart")
        st.line_chart(data)
        st.title("Area chart")
        st.area_chart(data)  
        Ytrain=np.ravel(ytrain)
        clf.fit(xtrain,ytrain)
        st.write("training complected")
        td=np.array([[30,140,85,7,98,70]])
        res=clf.predict(td)
        print(res)
        st.write(res[0])
        import joblib
        joblib.dump(clf,"Maternalmodeal.pk1")    

    elif  choice == "Charts":
        # Data visualization
        import numpy as np
        import pandas as pd
        st.title("Bar chart")
        data=pd.DataFrame(np.random.randn(50,2),columns=["DiastolicBP","BP"])
        st.bar_chart(data)
        st.title("Line chart")
        st.line_chart(data)
        st.title("Area chart")
        st.area_chart(data)  

 


        
         
if __name__== '__main__':
    main()
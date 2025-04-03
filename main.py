import numpy as np
import pandas as pd
# d={"Name":[],"Subject":[],"Date":[],"Time":[],"Present":[]}
data=pd.read_csv("data.csv")

def mark_attendace(name,subject,date,time,present):
    global data
    new_row = pd.DataFrame([[name, subject, date, time, present]], columns=data.columns)
    data = pd.concat([data, new_row], ignore_index=True)



mark_attendace("sakshi","dl"," 3 april","12:12:10 pm","no")

data.to_csv("data.csv",index=False)

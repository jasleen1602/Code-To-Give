import numpy as np
import pandas as pd
#import preprocessor as p
import counselor
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path
from PIL import Image
import streamlit as st
import imagify
from bokeh.plotting import figure, output_file, show
import math
from bokeh.palettes import Greens
from bokeh.transform import cumsum
from bokeh.models import LabelSet, ColumnDataSource
#ap = Path.joinpath(Path.cwd(), 'models')
#dsp = Path.joinpath(Path.cwd(), 'dataset')

#model = load_model(Path.joinpath(artifacts_path, 'botmodel.h5'))
#tok = joblib.load(Path.joinpath(artifacts_path, 'tokenizer_t.pkl'))
#words = joblib.load(Path.joinpath(artifacts_path, 'words.pkl'))
#df2 = pd.read_csv(Path.joinpath(datasets_path, 'bot.csv'))

model = load_model('botmodel.h5')
tok = joblib.load('tokenizer_t.pkl')
words = joblib.load('words.pkl')
df2 = pd.read_csv('bot.csv')
flag=1

import string
import re
import json
import nltk
#run on the first time alone :
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def main():

    lem = WordNetLemmatizer()
    n=1
    def tokenizer(x):
        tokens = x.split()
        rep = re.compile('[%s]' % re.escape(string.punctuation))
        tokens = [rep.sub('', i) for i in tokens]
        tokens = [i for i in tokens if i.isalpha()]
        tokens = [lem.lemmatize(i.lower()) for i in tokens]
        tokens = [i.lower() for i in tokens if len(i) > 1]
        return tokens

    def no_stop_inp(tokenizer,df,c):
        no_stop = []
        x = df[c][0]
        tokens = tokenizer(x)
        no_stop.append(' '.join(tokens))
        df[c] = no_stop
        return df

    def inpenc(tok,df,c):
        t = tok
        x = x = [df[c][0]]
        enc = t.texts_to_sequences(x)
        padded = pad_sequences(enc, maxlen=16, padding='post')
        return padded

    def predinp(model,x):
        pred = np.argmax(model.predict(x))
        return pred

    def botp(df3,pred):
        l = df3.user[0].split()
        if len([i for i in l if i in words])==0 :
            pred = 1
        return pred

    def botop(df2,pred):
        x2 = df2.groupby('labels').get_group(pred).shape[0]
        idx1 = np.random.randint(0,x2)
        op = list(df2.groupby('labels').get_group(pred).bot)
        return op[idx1]

    def botans(df3):
        tok = joblib.load('tokenizer_t.pkl')
        word = joblib.load('words.pkl')
        df3 = no_stop_inp(tokenizer, df3, 'user')
        inp = inpenc(tok, df3, 'user')
        pred = predinp(model, inp)
        pred = botp(df3, pred)
        ans = botop(df2, pred)
        return ans

    def get_text():
        x = st.text_input("You : ")
        x=x.lower()
        xx = x[:13]
        if(xx =="start my test"):
            global flag
            flag=0
        input_text  = [x]
        df_input = pd.DataFrame(input_text,columns=['user'])
        return df_input

    #flag=1
    qvals = {"Select an Option": 0, "Yes": 3, "No": 1, "Not sure": 2}
    st.title("Buzz Women Bot")
    banner=Image.open("img/21.png")
    st.image(banner, use_column_width=True)
    st.title("About us")
    st.write("Buzz Women is the global movement by and for women. We bring transformation within reach. And enable women to ignite their personal and collective power.")
    st.write("The outcomes of the trainings are concrete and life improving: women grow their confidence, generate more cash, deal with climate change, take better care of their families and influence their communities.")
    st.write("Every single day we witness female power changing society. We don’t empower women, they empower themselves. We set the process in motion, they take the lead. Involving men and children along the way.")
    st.write("Welcome to the our career counseling session. I am your personal bot. Ask any queries regarding starting a business in the text box below and hit enter. I will assist you with it.")

    df3 = get_text()
    if (df3.loc[0, 'user']==""):
        ans = "Hi, I'm Buzz Women Bot. \nHow can I help you?"

    elif(flag==0):
        #st.write(flag)
        ans = "Sure, good luck!"
    else:
        ans = botans(df3)

    st.text_area("Buzz Women Bot:", value=ans, height=100, max_chars=None)

    if(flag==0):
        #x=start_test()
        #st.text_area("confirm", value="starting test", height=100, max_chars=None)
        st.title("Career Analysis")
        #st.write("Would you like to begin with the test?")
        kr = st.selectbox("Would you like me to help you choose what you should do for a living?", ["Select an Option", "Yes", "No"])
        if (kr == "Yes"):
            kr1 = st.selectbox("Select your highest level of education",
                               ["Select an Option", "Illiterate", "Literate" ,"Grade 8", "Grade 10", "Grade 12", "Undergraduate"])

            #####################################  GRADE 10  ###########################################

            if(kr1!="Select an Option"):
                lis = []
                if (kr == "Yes"):
                    st.header("Question 1")
                    st.write("Do you have basic knowledge of fabric and style?")
                    n = imagify.imageify(n)
                    inp = st.selectbox("",
                                       ["Select an Option", "Yes", "No", "Not sure"],
                                       key='1')
                    if ((inp != "Select an Option")):
                        lis.append(qvals[inp])
                        st.header("Question 2")
                        st.write("Do you know how to stich, zigzag or staright?")
                        n = imagify.imageify(n)
                        inp2 = st.selectbox("", ["Select an Option", "Yes", "No", "Not sure"], key='2')

                        if (inp2 != "Select an Option"):
                            lis.append(qvals[inp2])
                            st.header("Question 3")
                            st.write("Do you know how to use a sewing machine?")
                            n = imagify.imageify(n)
                            inp3 = st.selectbox("", ["Select an Option", "Yes", "No", "Not sure"], key='3')
                            if (inp3 != "Select an Option"):
                                lis.append(qvals[inp3])
                                st.header("Question 4")
                                st.write("Do you know how to read an inch tape and take measurements?")
                                n = imagify.imageify(n)
                                inp4 = st.selectbox("",
                                                    ["Select an Option", "Yes", "No", "Not sure"], key='4')
                                if (inp4 != "Select an Option"):
                                    lis.append(qvals[inp4])
                                    st.header("Question 5")
                                    st.write("Do you know how to do waxing, threading and massaging?")
                                    n = imagify.imageify(n)
                                    inp5 = st.selectbox("",
                                                        ["Select an Option", "Yes", "No", "Not sure"], key='5')
                                    if (inp5 != "Select an Option"):
                                        lis.append(qvals[inp5])
                                        st.header("Question 6")
                                        st.write(
                                            "Have you dressed someone before for a function?")
                                        n = imagify.imageify(n)
                                        inp6 = st.selectbox("",
                                                            ["Select an Option", "Yes", "No", "Not sure"], key='6')
                                        if (inp6 != "Select an Option"):
                                            lis.append(qvals[inp6])
                                            st.header("Question 7")
                                            st.write(
                                                "Do you know how to do make up?")
                                            n = imagify.imageify(n)
                                            inp7 = st.selectbox("",
                                                                ["Select an Option", "Yes", "No", "Not sure"], key='7')
                                            if (inp7 != "Select an Option"):
                                                lis.append(qvals[inp7])
                                                st.header("Question 8")
                                                st.write(
                                                    "Can you can cook?")
                                                n = imagify.imageify(n)
                                                inp8 = st.selectbox("",
                                                                    ["Select an Option", "Yes", "No", "Not sure"], key='8')
                                                if (inp8 != "Select an Option"):
                                                    lis.append(qvals[inp8])
                                                    st.header("Question 9")
                                                    st.write(
                                                        "Do people love to eat your food?")
                                                    n = imagify.imageify(n)
                                                    inp9 = st.selectbox("",
                                                                        ["Select an Option", "Yes", "No", "Not sure"], key='9')
                                                    if (inp9 != "Select an Option"):
                                                        lis.append(qvals[inp9])
                                                        st.header("Question 10")
                                                        st.write(
                                                            "Do people suggest you to be a caterer")
                                                        n = imagify.imageify(n)
                                                        inp10 = st.selectbox("",
                                                                             ["Select an Option", "Yes", "No", "Not sure"], key='10')
                                                        if (inp10 != "Select an Option"):
                                                            lis.append(qvals[inp10])
                                                            st.header("Question 11")
                                                            st.write(
                                                                "Are you aware about the quality of vegetables?")
                                                            n = imagify.imageify(n)
                                                            inp11 = st.selectbox("",
                                                                             ["Select an Option", "Yes", "No", "Not sure"], key='11')
                                                            if (inp11 != "Select an Option"):
                                                                lis.append(qvals[inp11])
                                                                st.header("Question 12")
                                                                st.write(
                                                                    "Do you know how to make food presentable?")
                                                                n = imagify.imageify(n)
                                                                inp12 = st.selectbox("",
                                                                             ["Select an Option", "Yes", "No", "Not sure"], key='12')
                                                                if (inp12 != "Select an Option"):
                                                                    lis.append(qvals[inp12])
                                                                    st.header("Question 13")
                                                                    st.write(
                                                                        "Are you good at making mehndi designs?")
                                                                    n = imagify.imageify(n)
                                                                    inp13 = st.selectbox("",
                                                                             ["Select an Option", "Yes", "No", "Not sure"], key='13')
                                                                    if (inp13 != "Select an Option"):
                                                                        lis.append(qvals[inp13])

                                                                        st.header("Question 14")
                                                                        st.write(
                                                                            "Can you make your own henna paste?")
                                                                        n = imagify.imageify(n)
                                                                        inp14 = st.selectbox("",
                                                                             ["Select an Option", "Yes", "No", "Not sure"], key='14')
                                                                        if (inp14 != "Select an Option"):
                                                                            lis.append(qvals[inp14])
                                                        
                                                                            st.header("Question 15")
                                                                            st.write(
                                                                                "Do you have good marketing skills?")
                                                                            n = imagify.imageify(n)
                                                                            inp15 = st.selectbox("",
                                                                                ["Select an Option", "Yes", "No", "Not sure"], key='15')
                                                                            if (inp15 != "Select an Option"):
                                                                                lis.append(qvals[inp15])

                                                                                st.header("Question 16")
                                                                                st.write(
                                                                                    "Are you good at convincing people?")
                                                                                n = imagify.imageify(n)
                                                                                inp16 = st.selectbox("",
                                                                                    ["Select an Option", "Yes", "No", "Not sure"], key='16')
                                                                                if (inp16 != "Select an Option"):
                                                                                    lis.append(qvals[inp16])
                                                        
                                                                                    st.header("Question 17")
                                                                                    st.write(
                                                                                        "Are you emapthetic means you can can put yourself in your customers’ shoes and truly understand what they want?")
                                                                                    n = imagify.imageify(n)
                                                                                    inp17 = st.selectbox("",
                                                                                        ["Select an Option", "Yes", "No", "Not sure"], key='17')
                                                                                    if (inp17 != "Select an Option"):
                                                                                        lis.append(qvals[inp17])

                                         
                                                                                    st.success("Analysis Completed")
                                                                                    #st.write(lis)
                                                                                    st.title("RESULTS:")
                                                                                    df = pd.read_csv(r"occupation.csv")

                                                                                    input_list = lis

                                                                                    occupations = {1: "Tailoring",
                                                                                                    2: "Beautician",
                                                                                                    3: "Catering",
                                                                                                    4: "Mehndi Artist",
                                                                                                    5: "Cloth Seller"
                                                                                                }

                                                                                    def output(listofanswers):
                                                                                        class my_dictionary(dict):
                                                                                            def __init__(self):
                                                                                                self = dict()

                                                                                            def add(self, key, value):
                                                                                                self[key] = value

                                                                                        score = my_dictionary()
                                                                                        csum=0
                                                                                        for i in range(0,4):
                                                                                            csum+=input_list[i]
                                                                                        score.add(0, csum/4)
                                                                                        csum=0
                                                                                        for i in range(4,7):
                                                                                            csum+=input_list[i]
                                                                                        score.add(1, csum/3)
                                                                                        csum=0
                                                                                        for i in range(7,12):
                                                                                            csum+=input_list[i]
                                                                                        score.add(2, csum/5)
                                                                                        csum=0
                                                                                        for i in range(12,14):
                                                                                            csum+=input_list[i]
                                                                                        score.add(3, csum/2)
                                                                                        csum=0
                                                                                        for i in range(14,17):
                                                                                            csum+=input_list[i]
                                                                                        score.add(4, csum/3)

                                                                                        all_scores = []

                                                                                        for i in range(5):
                                                                                            all_scores.append(score[i])

                                                                                        li = []

                                                                                        for i in range(len(all_scores)):
                                                                                            li.append([all_scores[i], i])
                                                                                        li.sort(reverse=True)
                                                                                        sort_index = []
                                                                                        for x in li:
                                                                                            sort_index.append(x[1] + 1)
                                                                                        all_scores.sort(reverse=True)

                                                                                        a = sort_index[0:5]
                                                                                        b = all_scores[0:5]
                                                                                        s = sum(b)
                                                                                        d = list(map(lambda x: x * (100 / s), b))

                                                                                        return a, d

                                                                                    l, data = output(input_list)

                                                                                    out = []
                                                                                    for i in range(0, 5):
                                                                                        n = l[i]
                                                                                        c = occupations[n]
                                                                                        out.append(c)

                                                                                    output_file("pie.html")

                                                                                    graph = figure(title="Recommended career options", height=500,
                                                                                                width=500)
                                                                                    radians = [math.radians((percent / 100) * 360) for percent
                                                                                            in data]

                                                                                    start_angle = [math.radians(0)]
                                                                                    prev = start_angle[0]
                                                                                    for i in radians[:-1]:
                                                                                        start_angle.append(i + prev)
                                                                                        prev = i + prev

                                                                                    end_angle = start_angle[1:] + [math.radians(0)]

                                                                                    x = 0
                                                                                    y = 0

                                                                                    radius = 0.8

                                                                                    color = Greens[len(out)]
                                                                                    graph.xgrid.visible = False
                                                                                    graph.ygrid.visible = False
                                                                                    graph.xaxis.visible = False
                                                                                    graph.yaxis.visible = False

                                                                                    for i in range(len(out)):
                                                                                        graph.wedge(x, y, radius,
                                                                                                    start_angle=start_angle[i],
                                                                                                    end_angle=end_angle[i],
                                                                                                    color=color[i],
                                                                                                    legend_label=out[i] + "-" + str(
                                                                                                        round(data[i])) + "%")

                                                                                    graph.add_layout(graph.legend[0], 'right')
                                                                                    st.bokeh_chart(graph, use_container_width=True)
                                                                                    labels = LabelSet(x='text_pos_x', y='text_pos_y',
                                                                                                        text='percentage', level='glyph',
                                                                                                        angle=0, render_mode='canvas')
                                                                                    graph.add_layout(labels)

                                                                                    st.header('More information on each career option')
                                                                                    # We'll be using a csv file for that
                                                                                    for i in range(0, 5):
                                                                                        st.subheader(occupations[int(l[i])])
                                                                                        st.write(df['about'][int(l[i]) - 1])

                                                                                    st.header('Average Earning')
                                                                                    # We'll be using a csv file for that
                                                                                    for i in range(0, 5):
                                                                                        st.subheader(occupations[int(l[i])])
                                                                                        st.write(df['avgsal'][int(l[i]) - 1])

                                                            

if __name__=="__main__":
    main()                  
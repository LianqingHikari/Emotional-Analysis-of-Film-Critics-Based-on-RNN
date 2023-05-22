from tkinter import *
from tkinter import scrolledtext
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
import numpy as np


#-----------------------基本界面------------------------
#创建窗口
window = Tk()
window.title("电影影评情感识别")#设置标题
window.geometry("350x300")#设置大小
window.minsize(350,300)#固定窗口大小
window.maxsize(350,300)


#添加标签
lbl_input_text = Label(window, text="请输入影评文本:",font=("微软雅黑",12))#输入文本标签
lbl_input_text.place(x=10,y=10)#设置位置

lbl_output_emotion = Label(window, text="情感：",font=("微软雅黑",12))#情感标签
lbl_output_emotion.place(x=25,y=230)
lbl_output_emotion_show = Label(window, text=" ")
lbl_output_emotion_show.place(x=65,y=230)

lbl_output_score = Label(window, text="得分：",font=("微软雅黑",12))#得分标签
lbl_output_score.place(x=25,y=260)
lbl_output_score_show = Label(window, text=" ")
lbl_output_score_show.place(x=65,y=260)

#添加文本区
txt_input_text = scrolledtext.ScrolledText(window, width=40, height=10)#输入区域
txt_input_text.place(x=25,y=40)
txt_input_text.focus()#设置焦点




#-----------------------使用模型进行预测------------------------
NATURE_NUM=2.7182818284#设置常数

#载入已训练好的模型
model = keras.models.load_model('my_model.h5')

#载入编码器
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)

encoder = info.features['text'].encoder

#输出函数
def sigmoid(activa_data):#sigmoid归一化函数
    temp_num=1/(1+NATURE_NUM**(-activa_data))
    lbl_output_score_show.configure(text="%.2f"%(temp_num*10))
    if(temp_num>=0.5):
        lbl_output_emotion_show.configure(text="积极")
    else:
        lbl_output_emotion_show.configure(text="消极")
    return temp_num
    

def predict(str_a):
    str_b="cmy is nb"#辅助字符串，无意义
    encoded_str_a=encoder.encode(str_a)
    encoded_str_b=encoder.encode(str_b)
    encoded_str_c=[encoded_str_a,encoded_str_b]
    padded_str_c=train_data = keras.preprocessing.sequence.pad_sequences(encoded_str_c,
                                                                         padding='post',
                                                                         maxlen=encoder.vocab_size)
    predict_result=model(padded_str_c)
    preditc_result_array=predict_result.numpy()
    sigmoid(preditc_result_array[0])

def start_predict():#按钮的应答
    str_predict=txt_input_text.get(1.0,END)
    predict(str_predict)

#添加按钮
btn_input = Button(window, text="确定",command=start_predict)#设置按钮的标签，响应函数
btn_input.place(x=260,y=180,width=60,height=35)
    

window.mainloop()

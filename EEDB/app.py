#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   app.py    
@Contact :   htkstudy@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/18 15:29   Armor(htk)     1.0         None
'''

import os
import pandas as pd
from flask import Flask,render_template

app = Flask(__name__,template_folder='templates',static_folder="templates",static_url_path='')

def get_data():
    filepath = "./dataDaily/"
    filename = [i for i in os.listdir("dataDaily")]
    # 按生成时间排序
    filename = sorted(filename, key=lambda x: os.path.getmtime(os.path.join(filepath, x)))
    last_file = os.path.join(filepath,filename[-1])
    print(last_file)
    data = pd.read_excel(last_file)
    data.sort_values(by="time",ascending=False,inplace=True)
    data.reset_index(drop=False,inplace=True)
    top20data = data.loc[:29,:]
    return top20data

@app.route('/',methods=['GET','POST'])
def index():
    set_refresh = 1
    top20data = get_data()
    return render_template('index.html',data=top20data,stop=set_refresh)

if __name__ == '__main__':
    app.run()
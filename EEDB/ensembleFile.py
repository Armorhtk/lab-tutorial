#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ensembleFile.py    
@Contact :   htkstudy@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/9 2:11   Armor(htk)     1.0         None
'''

import os
import datetime
import pandas as pd
import sys
sys.setrecursionlimit(10000)

def ensemble_your_file(file_path):
    file_name = [i for i in os.listdir(file_path)]
    print("files:",file_name)
    all_data = pd.DataFrame([])
    for fname in file_name:
        data = pd.read_excel(os.path.join(file_path,fname))
        all_data = pd.concat([all_data,data],axis=0)
        all_data.drop_duplicates("title",inplace=True)
        all_data.reset_index(drop=True,inplace=True)
    print("ensemble_datas:",all_data.shape[0])
    return all_data

if __name__ == '__main__':

    flag = 1
    if flag == 1:
        key = "新闻"
    elif flag == 2:
        key = "国家发布"
    file_dict = {"新闻":"dataDaily",
                 "国家发布":"chinaEmergency",}
    file_path = file_dict[key]
    all = ensemble_your_file(file_path)
    # all.sort_values("time", inplace=True)
    # all.reset_index(drop=True,inplace=True)
    # print(all[["time"]])

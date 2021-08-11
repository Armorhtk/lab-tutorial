#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   SinaNewsGet.py.py
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/17 13:40   Armor(htk)     1.0         None
'''

import os
import random
import pickle
import datetime
import requests
import pandas as pd
import time
from tqdm import tqdm
from bs4 import BeautifulSoup
from gne import GeneralNewsExtractor
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from ensembleFile import ensemble_your_file
import sys
sys.setrecursionlimit(10000)

# 单条详细新闻抽取
def SingleNewsExtractor(url,verbose=False):
    """
    url:新闻链接
    verbose:是否开启打印，默认为False
    """
    extractor = GeneralNewsExtractor()
    user_agent_pc = [
        # 谷歌
        'Mozilla/5.0.html (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.html.2171.71 Safari/537.36',
        'Mozilla/5.0.html (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.html.1271.64 Safari/537.11',
        'Mozilla/5.0.html (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.html.648.133 Safari/534.16',
        # 火狐
        'Mozilla/5.0.html (Windows NT 6.1; WOW64; rv:34.0.html) Gecko/20100101 Firefox/34.0.html',
        'Mozilla/5.0.html (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10',
        # opera
        'Mozilla/5.0.html (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.html.2171.95 Safari/537.36 OPR/26.0.html.1656.60',
        # qq浏览器
        'Mozilla/5.0.html (compatible; MSIE 9.0.html; Windows NT 6.1; WOW64; Trident/5.0.html; SLCC2; .NET CLR 2.0.html.50727; .NET CLR 3.5.30729; .NET CLR 3.0.html.30729; Media Center PC 6.0.html; .NET4.0C; .NET4.0E; QQBrowser/7.0.html.3698.400)',
        # 搜狗浏览器
        'Mozilla/5.0.html (Windows NT 5.1) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.html.963.84 Safari/535.11 SE 2.X MetaSr 1.0.html',
        # 360浏览器
        'Mozilla/5.0.html (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.html.1599.101 Safari/537.36',
        'Mozilla/5.0.html (Windows NT 6.1; WOW64; Trident/7.0.html; rv:11.0.html) like Gecko',
        # uc浏览器
        'Mozilla/5.0.html (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.html.2125.122 UBrowser/4.0.html.3214.0.html Safari/537.36',
    ]
    user_agent = {'User-Agent':random.choice(user_agent_pc)}
    rep = requests.get(url, headers=user_agent)
    source = rep.content.decode("utf-8",errors='ignore')
    result = extractor.extract(source)
    if verbose:
        print(result)
    return result

# 文件详细新闻抽取
def FileNewsExtractor(csv_file,save_file,verbose=False):
    """
    csv_file:csv路径
    save_file:保存文件路径
    verbose:是否开启打印，默认为False
    函数输入:csv文件路径（必须有url）、提取新闻的文件保存路径
    函数输出：具有详细新闻信息的pandas表
    """
    news = pd.DataFrame([],columns=['title', 'author', 'publish_time', 'content', 'images'])
    data = pd.read_excel(csv_file)
    for idx,news_url in tqdm(enumerate(data["url"]),total=len(data["url"])):
        news_infos = SingleNewsExtractor(news_url,verbose=verbose)
        news.loc[idx] = news_infos
        if idx % 3 and idx != 0:
            time.sleep(random.randint(1,3))
    news.to_excel(save_file,index=False)
    return news

# 主程序 - 测试
def test(c1 = False ,c2 = False):
    # 测试1
    if c1:
        df,day_html = SinaNewsExtractor(page_nums=50,stop_time_limit=5,verbose=1,withSave=True)
        print(df)
    # 测试 2
    if c2:
        FileNewsExtractor("dataDaily/2021_7_17_9_26news.xlsx","detailed_news.xlsx")

# 新浪新闻获取
def SinaNewsExtractor(page_nums=50,stop_time_limit=3,verbose=1,withSave=False):
    """
    url:爬取链接,具有既定格式 https://news.sina.com.cn/roll/#pageid=153&lid=2970&k=&num=50&page={}
    page_nums：爬取滚动新闻的页码数，可取值范围为[1,50]的整数,默认为50（最大值）
    stop_time_limit：为防止爬虫封锁IP,采用时停策略进行缓冲,可控制时停的上界,输入为一个整数，默认为3
    verbose:控制爬取可视化打印的标志位,0表示不显示，1表示html打印，2或其他数字表示详细打印，默认为1
    withSave：是否保存输出到当前文件夹下，默认False
    函数输入：url
    函数输出：具有新闻信息的pandas表格、由bs4组成的html列表
    """
    duplicates = False
    day_html = []
    filename = [i for i in os.listdir("dataDaily")]
    filename = sorted(filename, key=lambda x: os.path.getmtime(os.path.join("dataDaily", x)))
    history = pd.read_excel(os.path.join("dataDaily",filename[-1]))
    print(filename[-1])
    history_keys = history["title"].values
    df = pd.DataFrame([],columns=["label","title","time","url"])
    url_format = "https://news.sina.com.cn/roll/#pageid=153&lid=2970&k=&num=50&page={}"
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome("chromedriver.exe",chrome_options=chrome_options) # 启动driver
    driver.implicitly_wait(60)
    # 获取html，主要耗时
    for page in range(1,page_nums+1):
        driver.get(url_format.format(page))
        driver.refresh()
        soup = BeautifulSoup(driver.page_source,"lxml")
        frame = soup.find("div",attrs={"class":"d_list_txt","id":"d_list"})
        time.sleep(random.randint(1,stop_time_limit))
        if verbose != 0:
            print(page,url_format.format(page),len(day_html),duplicates)
        # 提取新闻信息，超快
        for li in frame.find_all("li"):
            url = li.a["href"]
            label = li.span.text
            title = li.a.text
            public = li.find("span",attrs={"class":"c_time"}).text
            # 判断重复
            if title in history_keys and duplicates != True:
                duplicates = True
                print("检测到重复爬取，已停止后续爬取")
                break
            df.loc[len(df)] = [label,title,public,url]
            if verbose == 2:
                print("{}\t{}\t{}".format(df.shape[0],public,title))
        if duplicates:
            break
    # 关闭driver，防止后台进程残留
    driver.quit()
    # 开启保存
    if withSave and len(df)!=0:
        if os.path.isdir("dataDaily") is False:
            os.makedirs('dataDaily')
        curr = datetime.datetime.now()
        curr_excel = "dataDaily/{}_{}_{}_{}_{}news.xlsx".format(curr.year,curr.month,curr.day,curr.hour,curr.minute)
        historyfiles = ensemble_your_file('dataDaily')
        df = pd.concat([df,historyfiles],axis=0)
        df.reset_index(drop=True,inplace=True)
        df.to_excel(curr_excel,index=False)
    return df,day_html

# 定时抽取新闻数据
def solidtime_real_data(step_time,nums):
    for i in range(nums):
        _,_ = SinaNewsExtractor(page_nums=5, stop_time_limit=5, verbose=1, withSave=True)
        print(f"{i+1}/{nums} epoch is end~")
        time.sleep(step_time)

if __name__ == '__main__':
    step = 60 * 60
    nums = int(7 * 86400/step)
    solidtime_real_data(step,nums)
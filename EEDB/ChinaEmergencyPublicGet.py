#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ChinaEmergencyPublicGet.py
@Contact :   htkstudy@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/8 22:38   Armor(htk)     1.0         None
'''

import os
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import requests
import time
import random
import datetime
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

# 网页处理函数
def dealHtml(html, verbose=False,withDuplicates=True):
    """
    html:html格式的输入
    verbose:控制爬取可视化打印的标志位,0表示不显示，1表示html打印，2表示详细打印，默认为1
    函数输入：html
    函数输出：具有新闻信息的pandas表格
    """
    duplicates_flag = False
    histoty_path = os.path.join("chinaEmergency",os.listdir("chinaEmergency")[-1])
    history = pd.read_excel(histoty_path)
    history_key = history["public_time"].sort_values(ascending=False).values[0]
    print("历史爬取节点：",history_key)
    soup = BeautifulSoup(html)
    tbody = soup.find("div", attrs={"class": "list_content"})
    for idx, tr in enumerate(tbody.find_all("li")):
        # 设置随机睡眠，防止封IP
        if random.randint(1, 2) == 2:
            time.sleep(random.randint(3, 6))
        # 如果首次爬取，则创建pandas表单
        if idx == 0:
            a = tr.find("a")
            title = a.text
            href = "http://www.12379.cn" + a["href"]
            content = contentExtractor(href)
            public_time = tr.find("span").text
            events_data = pd.DataFrame([], columns=["title", "content", "public_time", "urls"])
            events_data.loc[len(events_data), :] = [title, content, public_time, href]
        else:
            # 非首次填充
            a = tr.find("a")
            title = a.text
            href = "http://www.12379.cn" + a["href"]
            content = contentExtractor(href)
            public_time = tr.find("span").text
            events_data.loc[len(events_data), :] = [title, content, public_time, href]
        if verbose:
            print([public_time,title, content,href])
        # 查重
        if withDuplicates:
            if history_key in  events_data["public_time"].values:
                events_data = events_data[:-1]
                duplicates_flag  = True
                print("监测到重复爬取，已停止后续爬取行为")
                break
    return events_data,duplicates_flag

# 正文抽取函数
def contentExtractor(url):
    """
    函数输入：url
    函数输出：输出当前url下的新闻正文内容，若没有新闻内容则输出"无"
    """
    # 先检测是否404
    user_agent = {
        'User-Agent': 'Mozilla/5.0.html (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.html.2171.71 Safari/537.36'}
    r = requests.get(url, headers=user_agent, allow_redirects=False)
    if r.status_code == 404:
        return "无"
    else:
        # 没有404，则抽取网页正文内容
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        driver = webdriver.Chrome("chromedriver.exe", chrome_options=chrome_options)
        driver.get(url)
        WebDriverWait(driver, 1000).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'content_text'))
        )
        respond = driver.page_source
        driver.quit()
        soup = BeautifulSoup(respond)
        text = soup.find("div", attrs={"class": "content_text"}).text
        return text

# 爬取器
class EventExtractor():
    def __init__(self):
        options = webdriver.ChromeOptions()
        options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        self.browser = webdriver.Chrome("chromedriver.exe", options=options,chrome_options=chrome_options)
        self.url = 'http://www.12379.cn/html/gzaq/fmytplz/index.shtml'
    # 登录
    def login(self):
        self.browser.get(self.url)
        WebDriverWait(self.browser, 1000).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'main'))
        )
    # 获取当前网页资源
    def get_html(self):
        return self.browser.page_source
    # 跳转下一页
    def next_page(self):
        try:
            submit_page = self.browser.find_element_by_xpath(r"//*[@class='next']")
            submit_page.click()
            WebDriverWait(self.browser, 1000).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'main'))
            )
            return self.browser.find_element_by_xpath(r"//*[@class='avter']").text
        except:
            print("页面挑战异常结束")
            return 0
    # 关闭driver
    def close(self):
        self.browser.close()

# 可控制抽取器
def ChinaEmergencyNewsExtractor(page_nums=3,verbose=1,withSave=False):
    """
    url:爬取链接,具有既定格式 https://news.sina.com.cn/roll/#pageid=153&lid=2970&k=&num=50&page={}
    page_nums：爬取滚动新闻的页码数，可取值范围为[1,25]的整数,默认为3
    verbose:控制爬取可视化打印的标志位,0表示不显示，1表示html打印，2表示详细打印，默认为1
    withSave：是否保存输出到当前文件夹下，默认False
    函数输入：url
    函数输出：保存实时爬取文件
    """
    temp = pd.DataFrame([], columns=["title", "content", "public_time", "urls"])
    event = EventExtractor()
    event.login()
    for i in range(0, page_nums):
        html = event.get_html()
        data,duplicates_flag = dealHtml(html, verbose=True)
        temp = pd.concat([temp, data], axis=0)
        temp.reset_index(drop=True, inplace=True)
        if duplicates_flag:
            break
        curr_page = event.next_page()
        if verbose and curr_page != 0:
            print(int(curr_page) - 1, temp.shape)
    # 关闭driver，防止后台进程残留
    event.close()
    # 开启保存
    if withSave and len(temp) != 0:
        if os.path.isdir("chinaEmergency") is False:
            os.makedirs('chinaEmergency')
        curr = datetime.datetime.now()
        curr_excel = "chinaEmergency/chinaEEP{}{}{}.xlsx".format(curr.year,curr.month,curr.day)
        temp.to_excel(curr_excel,index=False)
    return temp

# 定时抽取
def solidtime_real_data(step_time,nums):
    for i in range(nums):
        _ = ChinaEmergencyNewsExtractor(page_nums=3, verbose=1, withSave=True)
        print(f"{i+1}/{nums} epoch is end~")
        time.sleep(step_time)

if __name__ == '__main__':
    step = 60 * 60 * 24  # 每个x秒进行一次爬取
    nums = int(30 * 86400/step) # a * 86400 / step  a为天数；86400/step是自动计算当天爬取次数；可重写
    solidtime_real_data(step,nums)



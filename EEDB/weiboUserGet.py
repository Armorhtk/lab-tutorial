#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   weiboUserGet.py    
@Contact :   htkstudy@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/9 15:17   Armor(htk)     1.0         None
'''

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from tqdm import tqdm
import datetime
import time
import pickle
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

class WeiboUserExtractor():
    def __init__(self):
        #         options = webdriver.ChromeOptions()
        #         options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})
        self.browser = webdriver.Chrome("chromedriver.exe")
        self.url = 'https://weibo.com/chinanewsv'

    def login(self):
        self.browser.get(self.url)
        WebDriverWait(self.browser, 1000).until(
            EC.presence_of_element_located(
                (By.CLASS_NAME, 'pf_username')
            )
        )

    def get_html(self):
        htmls = []
        print("防封禁启动中，请在1分钟内扫二维码登录页面")
        for _ in tqdm(range(60)):
            time.sleep(1)
        for _ in range(0, 3):
            time.sleep(10)
            self.browser.execute_script("window.scrollTo(0,document.body.scrollHeight)")
            js = """
                 elements = document.getElementsByClassName('xxx');
                 for (var i=0; i < elements.length; i++)
                 {
                     elements[i].style.display='block'
                 }
                 """
            self.browser.execute_script(js)

            nodes = self.browser.find_elements_by_css_selector('.detail_wbtext_4CRf9')
            # 对每个节点进行循环操作
            for i in range(0, len(nodes), 1):
                try:
                    open_content = nodes[i].find_element_by_tag_name("span")
                    open_content.click()
                except:
                    pass
            htmls.append(self.browser.page_source)
            print("已共收集{}个HTML信息".format(len(htmls)))
            try:
                # 定位页面底部的一个标题
                self.browser.find_element_by_xpath(r'//*[@class="Bottom_text_1kFLe"]')
                # 如果没抛出异常就说明找到了底部标志，跳出循环
                break
            except Exception as e:
                # 抛出异常说明没找到底部标志，继续向下滑动
                pass
        print("爬取完毕,共收集{}个HTML信息".format(len(htmls)))
        return htmls

    def save_html(self, htmls):
        curr = datetime.datetime.now()
        curr_pkl = "weiboData/{}_{}_{}weibo.pkl".format(curr.year, curr.month, curr.day)
        pickle.dump(htmls, open(curr_pkl, "wb"))

    def close(self):
        self.browser.close()

if __name__ == '__main__':
    wb = WeiboUserExtractor()
    wb.login()
    htmls = wb.get_html()
    wb.save_html(htmls)
    wb.close()
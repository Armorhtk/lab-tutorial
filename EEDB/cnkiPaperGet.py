#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cnkiPaperGet.py    
@Contact :   htkstudy@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/9 13:46   Armor(htk)     1.0         None
'''


from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import random
import os
import warnings
import pandas as pd
warnings.filterwarnings("ignore")


def dealHtml(html, verbose=False):
    soup = BeautifulSoup(html)
    tbody = soup.find("table", attrs={"class": "GridTableContent"})
    for idx, tr in enumerate(tbody.find_all("tr")):
        if idx == 0:
            cols = [i.text.replace("\n", "") if i.text.replace("\n", "") is not "" else "序号" for i in tr.find_all("td")]
            cols.append("链接")
            cnki_data = pd.DataFrame([], columns=cols)
            if verbose:
                print(cols)
        else:
            row_content = [i.text.replace("\n", "").replace(" ", "") if i.text.replace("\n", "") is not "" else 0 for i
                           in tr.find_all("td")]
            row_content.extend(
                ["https://kns.cnki.net/kcms/" + td.find("a", attrs={"class": "fz14"})["href"][5:] for td in
                 tr.find_all("td") if td.find("a", attrs={"class": "fz14"}) is not None])
            cnki_data.loc[len(cnki_data), :] = row_content
            if verbose:
                print(row_content)
    return cnki_data

class CnkiExtractor():
    def __init__(self, key_word,subject="信息科技",open_self_choose=False):
        self.browser = webdriver.Chrome("chromedriver.exe")
        self.url = 'https://kns.cnki.net/kns/brief/result.aspx?dbprefix=CJFQ'
        self.count = 1
        self.key_word = key_word
        self.subject = subject
        self.open_self_choose = open_self_choose

    def login(self):
        self.browser.get(self.url)
        WebDriverWait(self.browser, 1000).until(
            EC.presence_of_element_located(
                (By.ID, 'txt_1_value1')
            )
        )

        # 是否开启自定义 选择 学科领域
        if self.open_self_choose:
            search = self.browser.find_element_by_xpath('//*[@id="txt_1_value1"]')
            search.send_keys(self.key_word)
            time.sleep(1*20)
        else:
            # 选择学科 先清除所有学科，勾选想要的学科
            js = 'document.getElementsByClassName("btn")[0].click();'
            self.browser.execute_script(js)
            subject_frame = self.browser.find_element_by_xpath(f'//*[@name="信息科技"]')
            subject_frame.send_keys(Keys.ENTER)
            # 检索主题 关键词 并点击
            search = self.browser.find_element_by_xpath('//*[@id="txt_1_value1"]')
            search.send_keys(self.key_word)
            submit_search = self.browser.find_element_by_id('btnSearch')
            submit_search.click()

        # 转换为 内嵌表单 选择所有中文文献
        self.browser.switch_to.frame('iframeResult')
        WebDriverWait(self.browser, 1000).until(
            EC.presence_of_element_located(
                (By.CLASS_NAME, 'Ch-En')
            )
        )
        submit_chinese = self.browser.find_element_by_link_text("中文文献")
        submit_chinese.click()
        time.sleep(2)
        # 点击 显示每页50篇论文
        submit_page = self.browser.find_element_by_xpath('//*[@id="id_grid_display_num"]/a[3]')
        submit_page.click()

    def get_info(self):
        # 打印出检索结果和页数
        submit_span_result = self.browser.find_element_by_xpath(r"//*[@class='pagerTitleCell']")
        submit_span_page = self.browser.find_element_by_xpath(r"//*[@class='countPageMark']")
        submit_span_result = submit_span_result.text.replace(" ", "")
        submit_span_page = submit_span_page.text[2:]
        print(f"登录成功,共检索{submit_span_result},共{submit_span_page}页")
        return int(submit_span_page)

    def get_html(self):
        return self.browser.page_source

    def next_page(self):
        try:
            submit_page = self.browser.find_element_by_xpath(
                r"//*[@class='TitleLeftCell']/font/following-sibling::a[1]")
            submit_page.click()
            return self.browser.find_element_by_xpath(r"//*[@class='TitleLeftCell']/font").text
        except:
            print("页面挑战异常结束")
            self.browser.close()

    def close(self):
        self.browser.close()


if __name__ == '__main__':
    key_words = "突发事件"
    save_file = f"CNKIDaily/{key_words }知网论文测试.xlsx"
    if os.path.exists(save_file):
        temp = pd.DataFrame([])
        temp.to_excel(save_file,index=False)
    temp = pd.read_excel(save_file)
    cnki = CnkiExtractor(key_words,open_self_choose=False)
    cnki.login()
    max_page = cnki.get_info()

    for i in range(0,max_page):
        html = cnki.get_html()
        data = dealHtml(html)
        temp = pd.concat([temp, data], axis=0)
        temp.reset_index(drop=True, inplace=True)
        temp.to_excel(save_file, index=False)
        curr_page = cnki.next_page()
        print(int(curr_page) - 1, temp.shape)
        time.sleep(random.randint(1, 3))
    cnki.close()
    temp.to_excel(save_file, index=False)


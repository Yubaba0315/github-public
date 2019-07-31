#!/usr/bin/python3   #为脚本语言指定使用python3解释器
# -*- coding: utf-8 -*-     #以utf-8格式编码，兼容中文字符
# @Author: civilchenyu@foxmail.com
# @File: lianjia_Spider.py

import requests
import sys
import importlib
from lxml import etree
from lxml.html import tostring
import re
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
importlib.reload(sys)
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")


# 爬取目标网站穷举
CITY_LIST = []
SEARCH_LIST = []
MAX_RETRY = 10
TIMEOUT = 30

# 定义被调用函数定义，类（class）
class CsLianJiaSpider():

    # 0-class类的初始化参数，__init__的第一个参数必须为self
    def __init__(self):
        print('Author: civilchenyu@foxmail.com')
        self.start_url = 'https://www.lianjia.com/city/'
        self.resultCount = 0
        self.turn_page_url = 'https://su.lianjia.com/xiaoqu/rs{}/'
        self.site_counts = 0
        self.detail_counts = 0
        self.court = ""
        self.headers = {
        'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
        'Accept-Encoding':'gzip, deflate, br',
        'Accept-Language':'zh,zh-CN;q=0.9,en;q=0.8',
        'Cache-Control':'max-age=0',
        'Connection':'keep-alive',
        'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36', }
        self.get_citylist()

# 1-获取全国-城市列表[CITY_LIST],获取区域地址存入[SEARCH_LIST]列表
    def get_citylist(self):
        global city_name
        response = requests.get(self.start_url, headers=self.headers)
        response.encoding = 'utf-8'
        content = response.text.encode('unicode-escape').decode('unicode-escape')
        if not content:
            print('获取全国城市列表失败')
            return None
        whole_list = etree.HTML(content)
        provinces = whole_list.xpath("//div[@class='city_list_section']/ul[@class='city_list_ul']/li")
        if not provinces:
            print("find no province")
            return
        for one_province_ul in provinces:
            ul_tree = etree.HTML(tostring(one_province_ul))
            lis = ul_tree.xpath("//div[@class='city_province']/ul/li")
            if not lis:
                print("find no city")
                return
            for citys_li in lis:
                citys_tree = etree.HTML(tostring(citys_li))
                one_city_url = citys_tree.xpath("//a/@href")[0]
                city_name = citys_tree.xpath("//a/text()")
                city_name = "" if not city_name else city_name[0]

                CITY_LIST.append(one_city_url)

                # # 获取城市名
                # mother_url = re.findall("(.*)xiaoqu.*", str(url))[0]
                # response_mother_url = requests.get(mother_url, timeout=TIMEOUT, headers=self.headers)
                # response_mother_url.encoding = 'unicode'
                # if not response_mother_url:
                #     print('城市名获取失败')
                #     return
                # city_content = response_mother_url.text.encode('unicode-escape').decode('unicode-escape')
                # if not city_content:
                #     return
                # city_tree = etree.HTML(city_content)
                # city = city_tree.xpath("//span[@class='exchange']/text()")
                # city = "" if not city else city[0]

                now = time.strftime("%Y-%m-%d", time.localtime())
                file_city = open(str(now) + str(city_name) + '-OutputData.txt', 'a', encoding='utf-8')
                file_city.write(
                    '''城市,区域,板块,小区名字,90天成交量,正在出租套数,地铁情况,二手房在售套数,二手房挂牌均价（元）,建成时间,楼型,物业费（元/平米/月）,物业公司,开发商,总栋数,总户数''')
                file_city.write('\n')
                file_city.close()


        print('全国城市地址：', CITY_LIST)
        print('全国城市地址抓取完毕')
        for city_url in CITY_LIST:
            self.get_SEARCH_LIST(city_url)
        print('全国数据抓取完毕，程序结束')


    def get_SEARCH_LIST(self,city_url):
        SEARCH_LIST = []
        xiaoqu_url = str(city_url)+'xiaoqu/'
        response = requests.get(xiaoqu_url, headers=self.headers)
        response.encoding = 'utf-8'
        content = response.text.encode('unicode-escape').decode('unicode-escape')
        if not content:
            print('获取城市小区页面失败')
            return None
        quyu_list = etree.HTML(content)
        quyu_a = quyu_list.xpath("//div[@data-role='ershoufang']/div/a")
        if not quyu_a:
            print(str(city_url)+'站点无行政区')
            return
        for one_bankuai_a in quyu_a:
            quyu = etree.HTML(tostring(one_bankuai_a))
            tail_quyu_url=quyu.xpath("//@href")[0]
            quyu_url= str(city_url)+ tail_quyu_url
            SEARCH_LIST.append(quyu_url)
            print(str(quyu_url)+'行政区网址已加入爬取列表')
        print('本城市所有行政区地址：', SEARCH_LIST)
        self.start(SEARCH_LIST)

# 2-调用多线程，逐级爬取
    def start(self,SEARCH_LIST):
        for home_page_url in SEARCH_LIST:       # 遍历爬取一级网页
            try:
                pages = self.get_pages(home_page_url)       # 调用函数2，抓取网页
            except Exception as e:
                print("error get pages {}".format(home_page_url))
                print(e)
                pages = None
            if not pages:
                continue
            print('当前区域网址为 {} ，总页数为 {}'.format(home_page_url, pages))
            with concurrent.futures.ThreadPoolExecutor(pages) as executor:
                for page in range(1, pages + 1):
                    next_page_url = '{}pg{}/'.format(home_page_url, page)
                    executor.submit(self.list_page_parse, next_page_url)
        print("本城市数据抓取完毕！开始下一个城市！")

# 3-校验待爬取网页，从列表页中获取总页数，若失败调用4重新获取
    def get_pages(self, home_page_url):
        """
        从列表页中获取当前站点总页数
        :param home_page_url:
        :return:
        """
        response = requests.get(home_page_url, headers=self.headers)
        response.encoding = 'utf-8'
        content = response.text
        if not content:
            print('{} 下载失败'.format(home_page_url))
            return None
        pages = self.re_find_one('"totalPage":(\d+),', content)     # 调用函数3
        if pages:
            return int(pages)
        else:
            print("not pages: {}".format(home_page_url))
 # 4-重新获取网页源码中的text属性
    def re_find_one(self, reg, text):
        pattern = re.compile(reg)
        result = pattern.findall(text)
        return result[0]

# 5-一级页面（列表页）爬取
    def list_page_parse(self, url):
        global qu, bankuai, name_dept, deal_90d, rent_num, subway, onsale_num, price_dept, city_name, mother_url
        # 获取城市名
        mother_url = re.findall("(.*)xiaoqu.*", str(url))[0]
        response_mother_url = requests.get(mother_url, timeout=TIMEOUT, headers=self.headers)
        response_mother_url.encoding = 'unicode'
        if not response_mother_url:
            print('城市名获取失败')
            return
        city_content = response_mother_url.text.encode('unicode-escape').decode('unicode-escape')
        if not city_content:
            return
        city_tree = etree.HTML(city_content)
        city_name = city_tree.xpath("//span[@class='exchange']/text()")
        city_name = "" if not city_name else city_name[0]

        print("多线程遍历各页码一级页面： {}".format(url))
        response = requests.get(url, timeout=TIMEOUT, headers= self.headers)
        response.encoding = 'unicode'
        if not response:
            print("{} response is none.".format(url))
            return
        # 将response（网页源码）转义，编码为unicode，再解码为unicode码,此时 content 为字符串！
        content = response.text.encode('unicode-escape').decode('unicode-escape')
        if not content:
            print("{} content is none.".format(url))    # 将格式化内容的位置用大括号{}占位，用format()函数对网址格式化
            return
        # etree：构造了一个XPath解析对象并对HTML文本进行自动修正，相当于对content进行格式化
        tree = etree.HTML(content)
        lis = tree.xpath("//ul[@class='listContent']/li[@class='clear xiaoquListItem']")
        if not lis:
            print("find no house_list")
            return
        for li in lis:
            li_tree = etree.HTML(tostring(li))      #
            url = li_tree.xpath("//div[@class='title']/a/@href")
            if not url:
                continue
            else:
                url = url[0]
            # 区域
            qu = li_tree.xpath("//div[@class='positionInfo']/a[position()=1]/text()")
            qu = "" if not qu else qu[0]
            # 板块
            bankuai = li_tree.xpath("//div[@class='positionInfo']/a[position()=2]/text()")
            bankuai = "" if not bankuai else bankuai[0]
            # 小区名字
            name_dept = li_tree.xpath("//div[@class='title']/a[position()=1]/text()")
            name_dept = "" if not name_dept else name_dept[0]
            # 90天成交
            msgdeal_90d = li_tree.xpath("//div[@class='houseInfo']/a[position()=1]/text()")
            msgdeal_90d = "" if not msgdeal_90d else msgdeal_90d[0]
            try:
                deal_90d = re.findall(".*成交(.*)套.*", str(msgdeal_90d))[0]
            except Exception:
                deal_90d = msgdeal_90d
            # 出租
            msgrent_num = li_tree.xpath("//div[@class='houseInfo']/a[position()=2]/text()")
            msgrent_num = "" if not msgrent_num else msgrent_num[0]
            try:
                rent_num = re.findall("(.*)套.*", str(msgrent_num))[0]
            except Exception:
                rent_num = msgrent_num
            # "在售二手房*套"
            onsale_num = li_tree.xpath("//div[@class='xiaoquListItemRight']/div[@class='xiaoquListItemSellCount']/a[@class='totalSellCount']/span/text()")
            onsale_num = "" if not onsale_num else onsale_num[0]
            # 二手房挂牌均价
            price_dept = li_tree.xpath("//div[@class='xiaoquListItemRight']/div[@class='xiaoquListItemPrice']/div[@class='totalPrice']/span/text()")
            price_dept = "" if not price_dept else price_dept[0]
            # 地铁信息
            subway = li_tree.xpath("//div[@class='tagList']/span/text()")
            subway = "非地铁沿线" if not subway else subway[0]

            data1 = [city_name,
                     qu,
                     bankuai,
                     name_dept,
                     deal_90d,
                     rent_num,
                     subway,
                     onsale_num,
                     price_dept,]
            # 跳转二级页面信息爬取
            self.level2_list_page_parse(url, data1)

# 6-二级页面（具体楼盘页）爬取，并输出文件
    def level2_list_page_parse(self, url, data1):
        # request：向目标网页发起访问请求，将网页源码赋值给response，并以utf-8编码
        # timeout：限制网络请求响应时间/ headers：请求头模拟
        response = requests.get(url, timeout=TIMEOUT, headers= self.headers)
        response.encoding = 'unicode'
        if not response:
            print("{} response is none.".format(url))
            return
        # 将response（网页源码）转义，编码为unicode，再解码为unicode码,此时 content 为字符串！
        content = response.text.encode('unicode-escape').decode('unicode-escape')
        if not content:
            print("{} content is none.".format(url))    # 将格式化内容的位置用大括号{}占位，用format()函数对网址格式化
            return

        # etree：构造了一个XPath解析对象并对HTML文本进行自动修正，相当于对content进行格式化
        tree = etree.HTML(content)

        # 小区均价
        avg_price = tree.xpath("//span[@class='xiaoquUnitPrice']/text()")
        avg_price = "" if not avg_price else avg_price[0]
        # 建筑年代
        msgbuild_time = tree.xpath(u"//div[@class='xiaoquInfoItem']/span[string()='建筑年代']/following-sibling::*[1]/text()")
        msgbuild_time = "" if not msgbuild_time else msgbuild_time[0]
        try:
            build_time = re.findall("(.*)建成.*", str(msgbuild_time))[0]
        except Exception:
            build_time = msgbuild_time
        # 建筑类型
        basic_profile = tree.xpath(u"//div[@class='xiaoquInfoItem']/span[string()='小区概况']/following-sibling::*[1]/text()|//div[@class='xiaoquInfoItem']/span[string()='建筑类型']/following-sibling::*[1]/text()")
        basic_profile = "" if not basic_profile else basic_profile[0]
        # 物业费
        msgproperty_fee = tree.xpath(u"//div[@class='xiaoquInfoItem']/span[string()='物业费用']/following-sibling::*[1]/text()")
        msgproperty_fee = "" if not msgproperty_fee else msgproperty_fee[0]
        try:
           property_fee = re.findall("(.*)元.*", str(msgproperty_fee))[0]
        except Exception:
            property_fee = msgproperty_fee
        # 物业公司
        lianjia_property = tree.xpath(u"//div[@class='xiaoquInfoItem']/span[string()='物业公司']/following-sibling::*[1]/text()")
        lianjia_property = "" if not lianjia_property else lianjia_property[0]
        # 开发商
        developer = tree.xpath(u"//div[@class='xiaoquInfoItem']/span[string()='开发商']/following-sibling::*[1]/text()")
        developer = "" if not developer else developer[0]
        # 总栋数
        msghouse = tree.xpath(u"//div[@class='xiaoquInfoItem']/span[string()='楼栋总数']/following-sibling::*[1]/text()")
        msghouse = "" if not msghouse else msghouse[0]
        try:
            house = re.findall("(.*)栋.*", str(msghouse))[0]
        except Exception:
            house = msghouse
        # 总户数
        msghouseholds = tree.xpath(u"//div[@class='xiaoquInfoItem']/span[string()='房屋总数']/following-sibling::*[1]/text()")
        msghouseholds = "" if not msghouseholds else msghouseholds[0]
        try:
            households = re.findall("(.*)户.*", str(msghouseholds))[0]
        except Exception:
            households = msghouseholds

        # print('''城市，区域,板块,小区名字,90天成交量,正在出租套数,地铁情况,二手房在售套数,二手房挂牌均价（元）,建成时间,楼型,物业费（元/平米/月）,物业公司,开发商,总栋数,总户数''')
        data2 = [build_time,
                basic_profile,
                property_fee,
                lianjia_property,
                developer,
                house,
                households,]
        print(data1 + data2)

        # 输出
        file_country = open(str(now) + ' -WholeCountry-OutputData.txt', 'a', encoding='utf-8')
        file_country.write(','.join(data1+data2)+'\n')
        file_country.close()

        file_city = open(str(now) + str(city_name) + '-OutputData.txt', 'a', encoding='utf-8')
        file_city.write(','.join(data1+data2)+'\n')
        file_city.close()
        # print(data1 + data2)


# 主 程 序
if __name__ == "__main__":
    now = time.strftime("%Y-%m-%d", time.localtime())
    file_country = open(str(now) + ' -WholeCountry-OutputData.txt', 'a', encoding='utf-8')
    file_country.write('''城市,区域,板块,小区名字,90天成交量,正在出租套数,地铁情况,二手房在售套数,二手房挂牌均价（元）,建成时间,楼型,物业费（元/平米/月）,物业公司,开发商,总栋数,总户数''')
    file_country.write('\n')
    file_country.close()
    worker = CsLianJiaSpider()

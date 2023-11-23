# 사용 모듈
import requests
import csv
import os
import urllib.request
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import time


# 클래스 생성
class Musinsa():
    # 생성자 정의
    def __init__(self, item : str, code : str,page : int = 1):
        # 크롤링 URL page number
        self.page = page
        
        # 크롤링 URL 카테고리 항목
        self.item = item
        
        # 상품 목록 코드
        self.code = code

    # 저장 폴더 생성
    def storage(self):
		# Python 실행 스크랩트 기준 
        path = os.getcwd()
		# 데이터 저장 폴더 경로
        self.data_storage = os.path.join(path, "data_storage/")
		# 이미지 저장 폴더 경로
        img_storage = os.path.join(self.data_storage, "images/")
		# 이미지_카테고리별 폴더 경로
        self.category_storage = os.path.join(img_storage, f"{self.item}/")
		# 디렉토리 체크 및 생성
        try:
            if not os.path.isdir(self.data_storage):
                os.makedirs(self.data_storage)
        except OSError:
            print("Error : Creating directory " + self.data_storage)
        try:
            if not os.path.isdir(img_storage):
                os.makedirs(img_storage)
        except OSError:
            print("Error : Creating directory " + img_storage)
        try:
            if not os.path.isdir(self.category_storage):
                os.makedirs(self.category_storage)
        except OSError:
            print("Error : Creating directory " + self.category_storage)


	# Parser 생성기
    def get_parser(self):
		# 크롤링 URL(후기 많은 순서)
        url = f"https://www.musinsa.com/categories/item/{self.code}?d_cat_cd={self.code}&brand=&list_kind=small&sort=emt_high&sub_sort=&page={self.page}&display_cnt=100&group_sale=&exclusive_yn=&sale_goods=&timesale_yn=&ex_soldout=&plusDeliveryYn=&kids=&color=&price1=&price2=&shoeSizeOption=&tags=&campaign_id=&includeKeywords=&measure="
        
		# User-Agent 인증 객체
        user_agent = UserAgent()
        
		# User-Agent 인증 객체 랜덤으로 크롤링 에러 방지
        headers = {"User-Agent" : user_agent.random}
        
		# 사이트 접속
        req = requests.get(url, headers = headers)
        
		# 정상 접속되는지 확인
        if req.status_code == 200:
			# soup 생성
            soup = BeautifulSoup(req.text, "html.parser")
            return soup

    # Data 저장 함수 -> 이후 SQL 코드 변경 가능
    def save_data(self, data):
        header = ["url", "brand", "name", "price"]
        # 처음 생성 시 col까지 같이 생성
        if not os.path.exists(f"{self.data_storage}data_{self.item}.csv"):
            with open(f"{self.data_storage}data_{self.item}.csv", 'a') as f:
                writer = csv.DictWriter(f, fieldnames = header)
                writer.writeheader()
                for info in data:
                    writer.writerow(info)
        else:
            with open(f"{self.data_storage}data_{self.item}.csv", 'a', newline = '') as f:
                writer = csv.DictWriter(f, fieldnames = header)
                for info in data:
                    writer.writerow(info)


    def get_data(self, soup):
        # 데이터 버퍼
        data = []

        # 데이터 크롤링
        urls = soup.select("#goods_list  div.li_inner > div.list_img > a")
        brands = soup.select("#goods_list  div.li_inner > div.article_info > p.item_title > a")
        names = soup.select("#goods_list  div.li_inner > div.article_info > p.list_info > a")
        prices = soup.select("#goods_list  div.li_inner > div.article_info > p.price")
        data_count = len(urls)
        for idx in range(data_count):
            home_url = urls[idx]["href"][17:]
            img_data = urls[idx].img["data-original"]
            img_name = home_url.replace("/", "_")
            urllib.request.urlretrieve(img_data, f"{self.category_storage}{img_name}.jpg")
            brand = brands[idx].text
            name = names[idx].text.strip()

            if prices[idx].find("del") != None:
                del_string = prices[idx].find("del").text
                chang_price = int(prices[idx].text.replace(del_string, "").replace(",", "").replace("원", ""))
            else:
                chang_price = int(prices[idx].text.replace(",", "").replace("원", ""))
            data.append({"url" : home_url, "brand" : brand, "name" : name, "price" : chang_price})

        return data

    def get_start(self):
            max_page = int(self.get_parser().select_one("#goods_list > div.boxed-list-wrapper > div.pagingNumber-box.box > span > span.totalPagingNum").text)
            if max_page > 10:
                max_page = 10
            print(f"{self.item} : {self.code} Start")

            while max_page:
                try:
                    soup = self.get_parser()
                    data = self.get_data(soup)
                    self.save_data(data)
                    print(f" 현제 : {self.page} page\n 진행률 : {self.page / 10 * 100:.2f}%\n")
                    self.page += 1
                    max_page -= 1
                except:
                    self.page += 1
                    continue
if __name__ == "__main__":

    header = {"top" : ["001006", "001004", "001005", "001010", "001002", "001003", "001001", "001011", "001013", "001008"], 
                  "outer" : ["002022", "002001", "002002", "002025", "002017", "002003", "002020", "002019" ,"002023", "002004", "002018", "002008", "002007", "002024", "002009", "002013", "002012", "002016", "002021", "002014", "002006", "002015"], 
                  "pants" : ["003002", "003007", "003008", "003004", "003009", "003005", "003010", "003011", "003006"],
                  "onepiece" : ["020006", "020007", "020008"],
                  "skirt" : ["022002", "022001", "022003"]}

    item = "top"
    start = time.time()
    for code in header[item]:
        crow = Musinsa(item = item, page = 1, code = code)
        crow.storage()
        crow.get_start()
    end = time.time()
    print("소요시간 : ", end - start)


from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import csv

class ListingScraper:
    def __init__(self):
        self.BRAND_URL = "https://www.ss.lv/lv/transport/cars/"
        self.driver = webdriver.Chrome()
        self.all_listing_data = []
        
    def get_category_data(self):
        self.driver.get(self.BRAND_URL)
        wait = WebDriverWait(self.driver, 10)

        h4_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "h4.category")))

        categories = []
        cat_cnt = 0 

        for h4 in h4_elements:    
            a_tag = h4.find_element(By.CSS_SELECTOR, 'a.a_category')
            href = a_tag.get_attribute('href')
            category_name = a_tag.text 

            span_tag = h4.find_element(By.CSS_SELECTOR, "span.category_cnt")
            count_text = span_tag.text.strip('()')
            count = int(re.search(r'\d+', count_text).group())
            
            categories.append({
                "name": category_name,  
                "href": href + "sell/page{}.html", ## For easiest iteration later based on routing observations
                "count": count
            })
            
            cat_cnt += 1
            if cat_cnt > 40: break
        
        return categories
            
                
    def get_total_listing_count(self, categories):
       categories = self.get_category_data()
       return sum([category["count"] for category in categories])
        
    def get_all_listings_data(self, categories):
        for category in categories:
            listing_count = category["count"] 
            link = category["href"] 
            total_pages = (listing_count // 30) + (listing_count % 30 > 0) 
            page_counter = 1
            
            for page_number in range(1, total_pages + 1):
                page_url = link.format(page_number)
                self.driver.get(page_url) 
                html = self.driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                car_listings = soup.find_all('tr')
                for listing in car_listings:
                    details = listing.find_all(['td'], class_=['msga2-o', 'msga2-r'])
                    if len(details) >= 5:
                        self.all_listing_data.append([
                        category["name"],
                        details[0].get_text(strip=True),
                        details[1].get_text(strip=True),
                        details[2].get_text(strip=True),
                        details[3].get_text(strip=True),
                        details[4].get_text(strip=True)
                        ])
                    
                page_counter += 1

        
    def write_listing_data_to_csv(self, fname = "data.csv"):
        with open('data.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(self.all_listing_data)
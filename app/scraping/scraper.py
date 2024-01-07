from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import csv
from typing import List, Dict
class ListingScraper:
    def __init__(self):
        self.CATEGORY_URL = "https://www.ss.lv/lv/transport/cars/"
        self.driver = webdriver.Chrome()
        self.all_listing_data = []
        
    def get_category_data(self) -> List[Dict]:
        """
        Scrapes the page in ss.lv that shows all car brands and corresponding current listing amount for each brand.
        Returns a list of dictionaries, each containing category (brand) name,
        url for that category and amount of listings.
        """
        self.driver.get(self.CATEGORY_URL_URL)
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
            
                
    def get_total_listing_count(self) -> int:
        """
        Calculates the total amount of car listings.

        Returns:
            int: Amount of car listings
        """
        categories = self.get_category_data()
        return sum([category["count"] for category in categories])

    def get_all_listings_data(self, categories:List[Dict]):
        """
        Appends data about each car listing to the list of all listings.
        Iteratively goes through static links that are kept for each car category/brand,
        and using that fact that navigation can be done by appending 1.html ... n.html to the end of the link, 
        effectively scrapes data about each category until increase in page number results in new page opening. 

        Args:
            categories (List[Dict]): Data about car listings
        """
        for category in categories:
            listing_count = category["count"] 
            link = category["href"] 
            ### Approximate number of pages for category given that each page contains 30 listings.
            total_pages = (listing_count // 30) + (listing_count % 30 > 0) 
            page_counter = 1
            
            for page_number in range(1, total_pages + 1):
                page_url = link.format(page_number)
                self.driver.get(page_url) 
                html = self.driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                car_listings = soup.find_all('tr') # Listings are stored in the table rows, 1 listing - 1 row
                for listing in car_listings:
                    details = listing.find_all(['td'], class_=['msga2-o', 'msga2-r']) # (model, engine, mileage, year, price) in order
                    if len(details) >= 5:
                        self.all_listing_data.append([
                        category["name"],
                        details[0].get_text(strip=True),
                        details[1].get_text(strip=True),
                        details[2].get_text(strip=True),
                        details[3].get_text(strip=True),
                        details[4].get_text(strip=True)
                        ])
                    
                page_counter += 1 ## Go to the next page

        
    def write_listing_data_to_csv(self, fname = "data.csv"):
        """
        Writes the content of all listing data to separate csv file.
        
        Args:
            fname (str, optional): Filename. Defaults to "data.csv".
        """
        with open(fname, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(self.all_listing_data)
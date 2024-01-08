"""
Script for scraping all car listing from the ss.lv portal and writing them row by row to csv file.
"""

from scraper import ListingScraper

scraper = ListingScraper()

categories = scraper.get_category_data()

listings = scraper.get_all_listings_data(categories)

scraper.write_listing_data_to_csv()

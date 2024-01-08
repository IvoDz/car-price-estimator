from scraper import ListingScraper

scraper = ListingScraper()

categories = scraper.get_category_data()

listing_cnt = scraper.get_total_listing_count()

listings = scraper.get_all_listings_data(categories)

scraper.write_listing_data_to_csv()
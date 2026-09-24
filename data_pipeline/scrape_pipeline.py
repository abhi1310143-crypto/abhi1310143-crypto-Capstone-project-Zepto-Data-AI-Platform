import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.parse import urljoin


BASE_URL = "https://books.toscrape.com/"
MIN_BOOKS = 70

def get_soup(url):
    """
    Download a webpage and return its BeautifulSoup object.
    Parameters
    ----------
    url : str
        URL of the webpage.
    Returns
    -------
    BeautifulSoup
        Parsed HTML document.
    Raises 
    ------------
    requests.RequestException
        if the HTTP request fails.
    """
    response= requests.get(url,timeout=10)
    #to know whether the error is occured due to request failed
    response.raise_for_status()
    return BeautifulSoup (response.content,"html.parser")

def get_categories():
    """
    Extract category names and URLs from the website.

    Returns 
    -------
    list of dict
      Each dictionary contains category name and URL.
    """
    soup = get_soup(BASE_URL)

    categories =[]
    # to know categories present in side navigation
    """CSS selector (div.side_categories ul li ul li a ) targets the nested category links
      nested category links inside a multi level  vetical navigation menu"""
    
    category_links = soup.select(
        "div.side_categories ul li ul li a"
        )
    for link in category_links:

        category_name = link.get_text(strip = True)

        category_url = urljoin(
            BASE_URL,
            link.get("href","")
            )
        categories.append({
            "name": category_name,
            "url": category_url
        })

    return categories

def scrape_category(category_name, category_url):
    """
    Scrape all the books from one category.

    Handles pagination automatically

    Parameters
    ----------
    category_name : str
       Name of category.
    category_url : str
       URL of the category page.


    Returns
    -------
    list of dict
       Scrapped book records.

    """
    books =[]

    current_url= category_url

    while current_url:

        print(f"Scraping:{current_url}")

        soup=get_soup(current_url)

        # To find all the books on current page
        book_containers = soup.select("article.product_pod")

        for book in book_containers:

            #-------------------------
            #Title
            #-------------------------
            title_tag = book.select_one("h3 a")

            title = title_tag["title"].strip()
            #-----------------------
            #Price
            #-----------------------
            price_tag = book.select_one("p.price_color")

            price =price_tag.get_text(strip=True)
            #-----------------------
            #Star rating
            #-----------------------
            rating_tag = book.select_one("p.star-rating") 

            #Example class:
            # class= "star -rating Three"

            rating_classes =rating_tag.get("class",[])

            if len(rating_classes) > 1:
                star_rating = rating_classes[1]
            else:
                star_rating ="Unknown"
            #-------------------------
            #Availability
            #-------------------------
            availability_tag = book.select_one(
                'p.instock.availability'
            )
            availability = availability_tag.get_text(
                " ",
                strip = True )
            #---------------------------
            #Store book
            # ---------------------------
             
            books.append({
            "title": title,
               "price":price,
               "star_rating": star_rating,
               "availability":availability,
               "category":category_name
               })

        
        # Find next page
        # ------------------------
        next_button =soup.select_one("li.next a")

        if next_button:

            next_url =next_button["href"]

            #Category pages use relative URLs such as
            #page-2.html

            if next_url.startswith("http"):
                current_url = next_url
            else:
                #Current cateory URl ends in something like:
                #category/books_1/index,html.
                #remove index.html and add page number.
                if "index.html" in current_url:
                    current_url= current_url.replace(
                        "index.html",
                        next_url
                    )
                else:
                    current_url=current_url.rsplit("/",1)[0]+"/"+next_url
        else:
            current_url = None
        #Small delay between requests between requests to prevent overloading servers and avoid getting blocked
        time.sleep(0.2)
    return books



def main():
    """
    Run the main book-scraping pipeline.

    This function retrives book categories,scrapes books from 
    each category,and collects all scraped books into a list.

    The scraping process stops only after:
    1. At least 70 books have been scraped.
    2. At least 3 categories have been scraped.

    The function also prints the scraping progess, including:
    -Total categories found.
    -Current category being scraped.
    -Number of books scraped from each category.
    -Total number of books collected.

    Returns:
        None: Executes the scraping pipeline and prints progress.
    """

    print("Getting categories...")

    categories = get_categories()

    print(f"Total categories found:{len(categories)}")
    #-------------------------
    # store all scraped books
    #-------------------------
    all_books =[]



    # count how many categories are scraped, to ensure at least 3 categories are scraped
    categories_scraped = 0
    #-----------------------
    # scrape categories
    #-----------------------
  
    for category in categories:

        print("\n"+"="*50)
        print(f"Category:{category['name']}")
        print("="*50)

        category_books= scrape_category(
            category["name"],
            category["url"]
            )

        #Add books from this category to the main list
        all_books.extend(category_books)

        #Increase category counter
        categories_scraped += 1

        print(
            f"Books scraped from {category['name']}:"
            f"{len(category_books)}"
        )
        print(
            f"Total books collected:"
            f"{len(all_books)}"
        )
        #--------------------------------------------------------------------------------------------
        # stop only when both conditions are met: at least 70 books and at least 3 categories scraped
        #--------------------------------------------------------------------------------------------
        if len(all_books)>= MIN_BOOKS and categories_scraped >= 3:
            break
#------------------------------------------
#Convert to DataFrame
# -----------------------------------------
    df =pd.DataFrame(all_books)

    print("\n scraping completed!")

    print(f"\nTotal rows:{len(df)}")

    print(
          f"Total categories scraped:"
          f"{categories_scraped}")
    
    print("\nBefore cleaning:")

    print(df.head())
    #-----------------
    #save raw data
    #-----------------
    df.to_csv(
        "raw_books.csv",
        index =False
        )
    print("\nsaved file:raw_books.csv")
if __name__ == "__main__":
    main()
    
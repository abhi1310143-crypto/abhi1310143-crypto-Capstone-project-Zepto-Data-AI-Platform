import pandas as pd
import re
from pathlib import Path

#-------------------------------------
# File Paths
#-------------------------------------

BASE_DIR = Path(__file__).parent

INPUT_CSV= BASE_DIR / "raw_books.csv"
OUTPUT_CSV = BASE_DIR / "clean_books.csv"

#------------------------------------------
# Required project-defined conversion rate
#------------------------------------------

# Fixed project-defined rate:
GBP_TO_INR = 105.50

#---------------------------------
# clean price
#---------------------------------

def clean_price(price):
    """
    Convert price text into a float.
    
    Example:
    £51.77-> 51.77

    Invalid values become NaN.
    """

    try:
        price = str(price).strip().replace("£", "")
        return float(price)
    except (ValueError, TypeError):
        return float("nan")
    
#---------------------------------
# Clean rating
#---------------------------------

def clean_rating(rating):
    """
    Convert text rating into integer.
    Example:
    Three -> 3

    Invaild values become NaN
    """

    star_rating= {
        "One": 1,
        "Two": 2,
        "Three":3,
        "Four":4,
        "Five":5
    }
    return star_rating.get(
        str(rating).strip())

#---------------------------------
# Clean availability
#---------------------------------

def clean_stock(availability):
   """
    Convert availability text into boolean.
    'In stock' becomes True.
    'Out of stock' becomes False.
    Invalid values become None.
    """

   try:
        text = str(availability).strip().lower()

        if text.startswith("in stock"):
            return True
        elif text.startswith("out of stock"):
            return False
        else:
            return None
   except (ValueError, TypeError):
        return None
   
#---------------------------------
# Clean title
#---------------------------------

def clean_title(title: str) -> str:
        """
    Cleans book titles by:
    - Removing any text inside parentheses ()
    - Removing dangling closing parentheses
    - Removing trailing slash-number or hash-number patterns (e.g. #3, #2))
        """
        # Remove text inside parentheses
        cleaned =re.sub(r"\([^)]*\)", "", title)

        #Remove dangling closing parentheses
        cleaned = re.sub(r"\)+$", "", cleaned)

        # Remove trailing patterns such as #3 or /3
        cleaned = re.sub(r"[#/]\d+\)?$", "", cleaned)
        return cleaned.strip()

# ------------------------------------- 
# Show null percentage 
# -------------------------------------

def show_null_percentage(df):
    """
    Display null counts and null percentages
    for the columns used in the cleaning pipeline.
    """
    required_columns = [ 
        "title",
        "price_gbp", 
        "rating",
        "in_stock", 
        "category" 
        ]
    null_count = df[required_columns].isna().sum()
    null_percentage = ( df[required_columns].isna().mean() * 100 ).round(2)
    null_summary = pd.DataFrame({ "null_count": null_count, "null_percentage": null_percentage })
    print("\nNull analysis before imputation:")
    print(null_summary)

def main():
    """
    Cleaning the scraped data from "raw_books.csv" and create converted columns.

    Numeric missing values are filled using median.
    Invalid rows with missing required text fields are dropped.
    Invalid availability values are treated as out of stock. 
    GBP is converted to INR using the fixed rate of 
    1 GBP = 105.50 INR.
    """

    df=pd.read_csv(INPUT_CSV)

    print("Raw data shape:",df.shape)

    #-----------------------------
    # Clean title
    #-----------------------------

    df["title"] = df["title"].apply(clean_title)

    #-----------------------------
    # Clean price
    #-----------------------------

    df["price_gbp"]=df["price"].apply(clean_price)

    #-----------------------------
    # Clean rating
    #-----------------------------

    df["rating"] = df["star_rating"].apply(clean_rating)

    #------------------------------
    # Clean availability
    #------------------------------

    df["in_stock"] = df["availability"].apply(clean_stock)

    #-----------------------------
    # Clean required text fields 
    #-----------------------------
    df["title"] = df["title"].astype(str).str.strip() 
    df["category"] = df["category"].astype(str).str.strip()

    #-----------------------------------------
    # Show null percentage before imputation
    #-----------------------------------------
    show_null_percentage(df)

    #---------------------------------------------
    # Drop rows with missing required text fields
    # --------------------------------------------
     
    df = df[ (df["title"] != "") & (df["category"] != "") ]

    #-----------------------------------------------
    # Median imputation for invalid/missing prices
    #-----------------------------------------------
    df["price_gbp"] = df["price_gbp"].fillna(
        df["price_gbp"].median()
    )

    #------------------------------------------------
    # Median imputation for invalid/missing ratings
    #------------------------------------------------
    df["rating"] = df["rating"].fillna(
        df["rating"].median()
    ).round().astype(int)


    # Treat unrecognized availability as out of stock
    df["in_stock"] = df["in_stock"].fillna(False).astype(bool)

    # Convert GBP to INR using the required fixed project rate
    ## 1 GBP = 105.50 INR
    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)


    df = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category"
        ]
    ]

    # Save the cleaned data to a new CSV file
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nCleaning completed!")
    print("Total books:", len(df))
    print("Saved file:", "clean_books.csv")

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()

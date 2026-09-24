import sqlite3
import pandas as pd

#-----------------------------
#File Paths
#-----------------------------

INPUT_FILE = "clean_books.csv"
DATABASE_FILE = "books.db"

#-------------------
#Create Database
#-------------------

def create_database():
    """
    Creating a SQLite database from the cleaned books CSV file.

    This function reads the book data from clean_books.csv,
    creates categories and books tables,inserts the data,
    and saves the database as books.db.

    The try block executes database creation operations.
    If an error occurs,the except block catches the error,
    rolls back the transaction, and prints the error message.

    Finally block closes the database connection,
    ensuring that the connection is closed after execution.
    
    The categories table stores unique book categories.
    The books table stores book details and links each book to its category
    using a foregin key.

    Returns:
       None: Creates the database and prints the insertion summary.

    Raises:
        FileNotFoundError:if the cleaned CSV file is missing.
        Exception: If an error occurs during database creation.
    """
    #------------------------------
    # Read cleaned CSV file
    #------------------------------
    
    df = pd.read_csv(INPUT_FILE)

    #------------------------------
    # Connect to SQLite database
    #------------------------------
    
    connection = sqlite3.connect(DATABASE_FILE)

    cursor = connection.cursor()

    try:

        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")

        # Drop old tables
        # This fixes old schema errors
        cursor.execute("DROP TABLE IF EXISTS books")
        cursor.execute("DROP TABLE IF EXISTS categories")

        # Create categories table
        cursor.execute("""
            CREATE TABLE categories (

                category_id INTEGER PRIMARY KEY,

                category_name TEXT UNIQUE NOT NULL

            )
        """)

        # Create books table
        cursor.execute("""
            CREATE TABLE books (

                book_id INTEGER PRIMARY KEY,

                title TEXT NOT NULL,

                price_gbp REAL,

                price_inr REAL,

                rating INTEGER,

                in_stock INTEGER,

                category_id INTEGER,

                FOREIGN KEY (category_id)
                    REFERENCES categories(category_id)

            )
        """)

        # Insert unique categories
        categories = df[["category"]].drop_duplicates()

        for category in categories["category"]:

            cursor.execute(
                """
                INSERT INTO categories (category_name)
                VALUES (?)
                """,
                (category,)
            )

        # Create category lookup
        category_lookup = {}

        rows = cursor.execute(
            "SELECT category_id, category_name FROM categories"
        ).fetchall()

        for category_id, category_name in rows:

            category_lookup[category_name] = category_id

        # Insert books
        for _, row in df.iterrows():

            category_id = category_lookup[row["category"]]

            cursor.execute(
                """
                INSERT INTO books (

                    title,
                    price_gbp,
                    price_inr,
                    rating,
                    in_stock,
                    category_id

                )

                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["title"],
                    row["price_gbp"],
                    row["price_inr"],
                    int(row["rating"]),
                    int(row["in_stock"]),
                    category_id
                )
            )

        # Save changes
        connection.commit()

        print("Database created successfully!")

        print(
            "Books inserted:",
            cursor.execute(
                "SELECT COUNT(*) FROM books"
            ).fetchone()[0]
        )

        print(
            "Categories inserted:",
            cursor.execute(
                "SELECT COUNT(*) FROM categories"
            ).fetchone()[0]
        )

    except Exception as error:

        connection.rollback()

        print("Error:", error)

    finally:

        connection.close()


if __name__ == "__main__":

    create_database()

import sqlite3
import pandas as pd


DATABASE_FILE = "books.db"
OUTPUT_FILE = "query_outputs.txt"


def main():
    """
    Execute SQL queries and reproduce JOIN operations using pandas.

    This function connects to the SQLite database and executes
    five SQL queries demonstrating SELECT,WHERE,ORDER BY,
    LIMIT,DISTINCT,IN and JOIN operations.

    THE Function also :
        -Reads SQl query results into pandas DataFrames.
        -Retrives books and categories from the database.
        -Reproduces the SQL JOIN using pandas merge().
        -Filters books with ratings 4 and 5
        -Sorts results by rating in descending order.
        -Compares SQL JOIN results with pandas merge results.
        -Saves query results and comparison results to an output file.
    Returns:
       None:Prints query results and saves them to OUTPUT_FILE.
    
    Raises:
      sqlite3.Error:If a database operation fails.
      FileNotFoundError:If the database file is missing.

    """

    connection = sqlite3.connect(DATABASE_FILE)

    # Enable foreign keys
    connection.execute("PRAGMA foreign_keys = ON")

    queries = {

        "Query 1 - SELECT and WHERE": """
            SELECT title, price_gbp, rating
            FROM books
            WHERE rating >= 4
        """,

        "Query 2 - ORDER BY": """
            SELECT title, price_gbp
            FROM books
            ORDER BY price_gbp DESC
        """,

        "Query 3 - LIMIT": """
            SELECT title, price_gbp
            FROM books
            LIMIT 10
        """,

        "Query 4 - DISTINCT": """
            SELECT DISTINCT category_name
            FROM categories
        """,

        "Query 5 - IN and JOIN": """
           SELECT
                b.title,
                b.rating,
                b.price_inr,
                c.category_name
            FROM books AS b
            JOIN categories AS c ON b.category_id = c.category_id
            WHERE b.rating IN (4, 5)
            ORDER BY b.rating DESC
            LIMIT 10
        """ 
    }

    output_file = open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    )

    for name, query in queries.items():

        print("\n" + name)
        print(query)

        result = pd.read_sql(
            query,
            connection
        )

        print(result)

        output_file.write("\n" + name + "\n")
        output_file.write(query + "\n")
        output_file.write(result.to_string(index=False))
        output_file.write("\n")

    # Read two query results into pandas
    print("\n--- pandas read_sql examples ---")

    high_rating_df = pd.read_sql(
        """
        SELECT title, rating
        FROM books
        WHERE rating >= 4
        """,
        connection
    )

    expensive_df = pd.read_sql(
        """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10
        """,
        connection
    )

    print("\nHigh-rated books:")
    print(high_rating_df.head())

    print("\nMost expensive books:")
    print(expensive_df.head())

    # Read source tables into pandas
    books_df = pd.read_sql(
        "SELECT * FROM books",
        connection
    )

    categories_df = pd.read_sql(
        "SELECT * FROM categories",
        connection
    )

    # Reproduce the SQL JOIN using pandas merge
    merged_df = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    """
   book_id is included only to break ties when multiple books share the
   same rating, so sorting stays consistent and reproducible every time
   (ascending id = earlier/lower id shown first among same-rating books).
    It's not needed in the final output, so it's dropped after sorting
    and filtering are done, leaving only the columns meant for display.

    """

    pandas_join_result = merged_df[
        [   "book_id",
            "title",
            "rating",
            "price_inr",
            "category_name"
        ]
    ] 
    pandas_join_result = pandas_join_result[
        pandas_join_result["rating"].isin([4, 5])
    ]

    pandas_join_result = pandas_join_result.sort_values(
        by=["rating","book_id"],
        ascending=[False, True]
    ).head(10)
    pandas_join_result = pandas_join_result.drop(columns=["book_id"])

    print("\nPandas merge JOIN result:")
    print(pandas_join_result)

    # SQL JOIN result
    sql_join_result = pd.read_sql(
        queries["Query 5 - IN and JOIN"],
        connection
    )

    sql_join_result = sql_join_result.reset_index(drop=True)
    pandas_join_result = pandas_join_result.reset_index(drop=True)

    print("\nSQL JOIN equals pandas merge:")

    print(
        sql_join_result.equals(pandas_join_result)
    )

    output_file.write(
        "\n\nSQL JOIN result:\n"
    )

    output_file.write(
        sql_join_result.to_string(index=False)
    )

    output_file.write(
        "\n\nPandas merge result:\n"
    )

    output_file.write(
        pandas_join_result.to_string(index=False)
    )

    output_file.write(
        "\n\nJOIN results equal: "
        + str(
            sql_join_result.equals(pandas_join_result)
        )
    )

    output_file.close()

    connection.close()

    print("\nAll queries completed!")
    print("Saved output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
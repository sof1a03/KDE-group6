import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

rating_df = pd.read_csv('Ratings.csv')

#displaying the first 5 rows
rating_df.head()

rating_df.info()

# checking null values
rating_df.isna().sum()

# checking for unique user ids and isbn values
print('Number of unique user ids is {} and ISBN no. is {}'.format(rating_df['User-ID'].nunique(), rating_df['ISBN'].nunique()))

"""This means that many users are buying multiple books. Also some books are really famous and hence are bought by multiple users.*


"""

# making all the ISBN no. uppercase
rating_df['ISBN'].apply(lambda x: x.upper())

# checking for duplicates
rating_df[rating_df.duplicated()].sum()

books_df=pd.read_csv('books_cleaned.csv', low_memory=False)
# lets see if all the books in rating_df are also in books_df
rating_df_new = rating_df[rating_df['ISBN'].isin(books_df['ISBN'])]
print('Shape of rating_df: {} and rating_df_new: {}'.format(rating_df.shape, rating_df_new.shape))

# book ratings
rating_df_new['Book-Rating'].value_counts().reset_index()

# most popular books
rating_df_new.groupby('ISBN')['Book-Rating'].count().reset_index().sort_values(by='Book-Rating', ascending=False)[:10]

explicit_rating = rating_df_new[rating_df_new['Book-Rating'] != 0]
implicit_rating = rating_df_new[rating_df_new['Book-Rating'] == 0]
print('Shape of explicit rating: {} and implicit rating: {}'.format(explicit_rating.shape, implicit_rating.shape))

# most purchased books including the implicitely rated books
rating_df_new.groupby('ISBN')['User-ID'].count().reset_index().sort_values(by='User-ID', ascending=False)[:10]['ISBN'].values

# getting the book names corresponding to these ISBNs
isbn_nums = ['0971880107', '0316666343', '0385504209', '0060928336',
       '0312195516', '044023722X', '0142001740', '067976402X',
       '0671027360', '0446672211']
books_df[books_df['ISBN'].isin(isbn_nums)]

# most popular explicitely rated books
explicit_rating.groupby('ISBN')['Book-Rating'].count().reset_index().sort_values(by='Book-Rating', ascending=False)[:10]

# getting the book names corresponding to these ISBNs
isbn_nums = ['0316666343', '0971880107', '0385504209', '0312195516', '0060928336']
books_df[books_df['ISBN'].isin(isbn_nums)]

rating_df.to_csv('ratings_cleaned.csv', index=False)
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# A function to get the missing values count and it's percentage
def missing_values(df):
  """
  Description : This function takes a data frame as an input and gives missing value count and its percentage as an output
  function_name : missing_values
  Argument : dataframe.
  Return : dataframe

  """
  miss = df.isnull().sum() # finding the missing values.

  per = df.isnull().mean() # finding mean/ Average of missing values.
  df = pd.concat([miss,per*100],keys = ['Missing_Values','Percentage'], axis = 1) # concatenating both of them using concat method of pandas module.
  return df # returning dataframe

# Load datasets
books = pd.read_csv("Books.csv")

"""Books are identified by their respective ISBN. In order to not loose any book, any invalid ISBNs have already been identified and will be fixed in the dataset.
Note that in the case of several authors, only the first is provided.
Even if URLs linking to cover images are also given, appearing in three different flavors (Image-URL-S, Image-URL-M, Image-URL-L), i.e., small, medium, large, we're gonna drop this information because not relevant to our study case.
"""

books.head()

books.info()

# Checking duplicates in datasets using duplicated method of dataframe.
print(f'''\t  Duplicates in books data is {books.duplicated().sum()}''')

books.describe()

#correcting the ISBN
merged_invalid_books = pd.read_csv("merged_invalid_books.csv", encoding='latin1')

# Create a dictionary to map invalid ISBNs to their correct ISBN_true values
isbn_correction_map = dict(zip(merged_invalid_books['ISBN'], merged_invalid_books['ISBN_true']))
# Replace the ISBN values in books using the mapping
books['ISBN'] = books['ISBN'].replace(isbn_correction_map) #this is made in a separate ipynb code

# Drop unused columns
books = books.drop(columns=['Image_URL_S', 'Image_URL_M', 'Image_URL_L'], axis=1) # these columns are not relevant for our reccomendation system

missing_values(books)

"""number of missing values for Book_author and Publisher is negegable --> let's see if we can correct it"""

# Checking  for  null value in book author
books[books['Book_Author'].isna()]

"""both of books when we look online have no author, to simplify the process we're gonna drop these values."""

books = books.dropna(subset=['Book_Author'])

books[books['Publisher'].isna()]

#Replacing NaNs with correct  values
books.loc[128890, 'Publisher'] = 'Mundania Press LLC'
books.loc[129037, 'Publisher'] = 'Bantam'

#insepcting the values in year of publication
books['Year_Of_Publication'].unique()

# correcting DK publishing error
books[books['Year_Of_Publication'] == 'DK Publishing Inc']

# on searching for these  books we came to know about its authors
#ISBN '078946697X'
books.loc[books.ISBN == '078946697X','Year_Of_Publication'] = 2000
books.loc[books.ISBN == '078946697X','Book_Author'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X','Publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '078946697X','Book_Title'] = "DK Readers: Creating the X - Men, How It All Began (Level 4: Proficient Readers)"

#ISBN '0789466953'
books.loc[books.ISBN == '0789466953','Year_Of_Publication'] = 2000
books.loc[books.ISBN == '0789466953','Book_Author'] = "James Buckley"
books.loc[books.ISBN == '0789466953','Publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953','Book_Title'] = "DK Readers: Creating the X - Men, How Comic Books Come to Life (Level 4: Proficient Readers)"

#checking the rows having 'Gallimard' as yearOfPublication
books.loc[books['Year_Of_Publication'] == 'Gallimard',:]

books.loc[books.ISBN=='2070426769','Year_Of_Publication']=2003
books.loc[books.ISBN=='2070426769','Book_Author']='Jean-Marie Gustave Le ClÃ?Â©zio'
books.loc[books.ISBN=='2070426769','Publisher']='Gallimard'
books.loc[books.ISBN=='2070426769','Book_Title']="Peuple du ciel, suivi de 'Les Bergers"

# changing dtype of year of publication
books['Year_Of_Publication'] =books['Year_Of_Publication'].astype(int)

# something is off about years of publication like:
books[(books['Year_Of_Publication'] > 0) & (books['Year_Of_Publication'] < 1800)]

#replacing with correct  values
books.loc[books.ISBN=='	9643112136','Year_Of_Publication'] = 2010
books.loc[books.ISBN=='964442011X', 'Year_Of_Publication'] = 1991

#Sustituting np.Nan in rows with year=0 or  greater than the current year,2024
books.loc[(books['Year_Of_Publication'] > 2024) | (books['Year_Of_Publication'] == 0),'Year_Of_Publication'] = np.NAN

# replacing NaN values with median value of Year_Of_Publication
books['Year_Of_Publication'].fillna(int(books['Year_Of_Publication'].median()), inplace=True)

books['Book_Author'].value_counts()

books['Publisher'].value_counts()

books.info()

books = books.applymap(lambda x: x.lower() if isinstance(x, str) else x)
books[books.duplicated(subset=['Book_Title', 'Book_Author'], keep=False)]

#Dropping the rows with the entire column values are duplicated
books.drop_duplicates(subset=['Book_Title', 'Book_Author'], keep='first', inplace=True)
books.reset_index(drop=True,inplace=True)

books[books['Book_Title']=='emma']

books.info()

books.to_csv('books_cleaned.csv', index=False)
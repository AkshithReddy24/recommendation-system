# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 14:26:30 2020

@author: hello
"""

#Importing the required Libraries
import pandas as pd        
import numpy as np 
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

#reading the two datasets
read_books= pd.read_csv("goodbooks.csv", encoding='utf_8')  #books dataset
ratings= pd.read_csv("goodratings.csv")    #ratings dataset

#dropping the columns we don't need
books= read_books.drop(['goodreads_book_id', 'best_book_id', 'work_id',
                             'books_count','isbn','isbn13','title','language_code',
                             'average_rating','ratings_count','work_ratings_count','work_text_reviews_count','ratings_1',
                             'ratings_2','ratings_3','ratings_4','ratings_5','image_url','small_image_url'], axis= 'columns')
#merging the two datasets
book_rating= pd.merge(ratings, books, on= 'book_id').fillna(0)
#fillna is used to fill the nan values with 0


#dropping off the duplicate values to get the final dataset
book_rating= book_rating.drop_duplicates(['user_id', 'original_title'])


#making user interaction matrix 
book_rating_user_pivot= book_rating.pivot(index= 'original_title', columns= 'user_id', values='rating').fillna(0)


#now as there are many empty values in pivot we need to convert it in csr matrix
book_matrix=csr_matrix(book_rating_user_pivot.values)
#print(book_matrix)

#applying nearest neighbors algorithm
nn= NearestNeighbors(algorithm= 'brute', metric= 'cosine')
nn.fit(book_matrix)

#recommendation
choose_book=np.random.choice(book_rating_user_pivot.shape[0])
distances,indices=nn.kneighbors(book_rating_user_pivot.iloc[choose_book,:].values.reshape(1,-1),n_neighbors=6)
 
#prints 5 books for recommendation 
for i in range(0,len(distances.flatten())):
    if (i==0):
        print("Book Recommendation for book ",book_rating_user_pivot.index[choose_book])
    else:
        print(i,".",book_rating_user_pivot.index[indices.flatten()[i]])
#end of code
        
        

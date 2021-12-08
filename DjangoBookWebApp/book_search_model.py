
# import tensorflow as tf
# from tensorflow.keras import layers
import os
import pandas as pd
import numpy
import random
from torch import nn
import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
import collections
import string
import io
import json
import codecs

from thefuzz import fuzz
from thefuzz import process
from gensim.models.word2vec import Word2Vec, KeyedVectors
import gensim.downloader as gensim_api
import nltk


class TransformerEmbed(torch.nn.Module):
    '''
    TransformerEncoder for word embeddings. Since the synthetic queries are generated with out any type of order,
    we don't use a positional encoding.
    '''
    def __init__(self, in_dim, out_dim, transformer_dim=512):
        super(TransformerEmbed, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=8,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.initial = nn.Linear(in_dim,transformer_dim)
        self.final = nn.Linear(transformer_dim, out_dim)
        self.float()
    def call(self, inputs):
        x = self.initial(inputs)
        x = self.transformer_encoder(x)
        x = torch.mean(x,dim=1)
        x = nn.Sigmoid()(x)
        x = self.final(x)

        return x
        
class VectorWrapper():
    '''
    class which wraps operations in the vector space
    so we never have to think about it in the main search class.
    '''
    def __init__(self,int_to_vector_dict, book_to_int, int_to_book):
        self.int_to_vector_dict = int_to_vector_dict
        self.book_to_int = book_to_int
        self.int_to_book = int_to_book
        
        #We are defining cosine similarity as the distance metric. This is just the normed dot product.
        self.scoring_fxn = nn.CosineSimilarity(dim=1, eps=1e-6) 
        self.vector_tensor = self.__get_vectors_as_tensors()
        
    def title_to_vector(self,title):
        book_int = self.int_to_book[title]
        
    #function for combing vectors into a tensor for fast distance calculation
    def __get_vectors_as_tensors(self):
        vector_tensor = []
        for i in range(len(self.book_to_int)):
            vector_tensor.append(self.int_to_vector_dict[i+1])
        
        return torch.tensor(vector_tensor)     
        
    def __compute_distance(self,vector1,vector2):
        if len(vector1.shape) == 1: 
            vector1 = torch.unsqueeze(vector1, dim = 0)
        
        if len(vector2.shape) == 1:
            vector2 = torch.unsqueeze(vector2, dim = 0)
        
        return self.scoring_fxn(vector1,vector2)
    
    def compute_nearest_booksV(self,vector,num):
        l = list(self.__compute_nearest_vectors(vector,num))
        return [self.int_to_book[str(int(i)+1)] for i in l]
        
    def compute_nearest_booksT(self,title,num):
        book_int = self.book_to_int[title]
        vector = torch.tensor(self.int_to_vector_dict[book_int])
        l = list(self.__compute_nearest_vectors(vector,num))
        return [self.int_to_book[str(int(i)+1)] for i in l]
        
    # stack the vector we're concerned with so that we can compute distances in parallel. Yay pytorch!     
    def __compute_nearest_vectors(self,vector,num):
        if len(vector.shape) == 1:
            vector = torch.unsqueeze(vector, dim = 0)
            
        __vector_stack = torch.cat([vector for _ in range(len(self.book_to_int))])
        
        scores = self.scoring_fxn(__vector_stack,self.vector_tensor)
        return torch.topk(scores,num,sorted=True)[1]
        
    def compute_matches_from_vector(self,vector):
        return self.__compute_nearest_vector(vector)
    
class SearchAlgo():
    '''class for handling logic of searches'''
    def __init__(self, book_vectors, w2v, query_model, fuzzy_cutoff, book_to_int):
        self.past_searches = []
        self.vector_wrapper = book_vectors
        self.fuzzy_cutoff = fuzzy_cutoff      
        self.titles = book_to_int.keys()
        self.book_to_int = book_to_int
        self.w2v = w2v
        self.query_model = query_model
        self.tokenizer = nltk.RegexpTokenizer(r"\w+")
        self.stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
        with open('DjangoBookWebApp/research/most_popular_title_in_order') as json_file:
            self.most_popular_title_in_order = json.load(json_file)
    def get_nearest_books(self,vector,num):
        return self.vector_wrapper.compute_nearest_booksV(vector,num)
    
    # This function handles queries which weren't matched with titles.
    # It first calls __parse_query which deals with cleaning up the query. 
    # This intails filtering out stop words as well as words which the word2vec model doesn't know about
    
    def query_to_language_model(self, query, num):
        query, success = self.__parse_query(query)
        if not success:
            return None, False
        
        
        query = torch.tensor([[self.w2v[q] for q in query]])
        vector = self.query_model.call(query)
        
        books = self.get_nearest_books(vector, num)
        return books, True
        
    def __parse_query(self, query):
        query = self.tokenizer.tokenize(query)
        query = [word.lower() for word in query if (word.lower() not in self.stopwords and word.lower() in self.w2v.key_to_index.keys())]
        
        return(query,len(query)>0)
    
    # checks to see for matches on titles. 
    # This would need more engineering to make sure your not matching when you dont want to be.
    def __check_titles(self,query):
        print(query)
        for title in self.most_popular_title_in_order:
            if type(title) == str:
                if query.lower() in title.lower():
                    return title
        else:
            return None
            
            
    # main method which mostly just handles the logic, nothing so special
    def make_query(self, query, num = 5):
        #no searching by author!
        title_match = self.__check_titles(query)
        if title_match:
            #print('title found')
            return self.vector_wrapper.compute_nearest_booksT(title_match, num)
            
        else:
            books,success = self.query_to_language_model(query, num)
            if not success:
                return ["You entered a confusing word! No good recomendations..."]
            return books
            
                
             
class FinalWrapper():
    def __init__(self):
        #raise NameError(os.getcwd())
        self.__init_files()
        self.w2v = KeyedVectors.load('DjangoBookWebApp/research/word2vec.wordvectors')
        self.stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
        self.vectorWrapper = VectorWrapper(self.int_to_weight,self.book_to_int,self.int_to_book)        
        self.search = SearchAlgo(self.vectorWrapper,self.w2v,self.transformer, .8, self.book_to_int)
        
        
    def __init_files(self):
    	
        with open('DjangoBookWebApp/research/most_popular_title_in_order') as json_file:
            self.most_popular_title_in_order = json.load(json_file)

        with open('DjangoBookWebApp/research/dict.json') as json_file:
            self.correct_word_dict = json.load(json_file)

        with open('DjangoBookWebApp/research/book_to_int.json') as json_file:
            self.book_to_int = json.load(json_file)

        with open('DjangoBookWebApp/research/int_to_book.json') as json_file:
            self.int_to_book = json.load(json_file)

        with open('DjangoBookWebApp/research/book_id_to_correct_title.json') as json_file:
            self.book_id_to_correct_title = json.load(json_file)

        with open('DjangoBookWebApp/research/incorrect_title_to_book_id.json') as json_file:
            self.incorrect_title_to_book_id = json.load(json_file)

        with open('DjangoBookWebApp/research/int_to_weight.json') as json_file:
            temp = json.load(json_file)
            
        self.int_to_weight = {int(k):np.asarray(v) for k,v in temp.items()}
        self.transformer = transformer = TransformerEmbed(100,80)
        self.transformer.load_state_dict(torch.load('DjangoBookWebApp/research/query_model'))
        
    def GetRecs(self, query):
        return self.search.make_query(query, 6)

# fr = FinalWrapper()
# print(fr.GetRecs("snakes and air"))
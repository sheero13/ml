#txtfile
# import re
# import string
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from gensim.models import Word2Vec

# # Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Define stopwords, stemmer, and lemmatizer
# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()

# # Preprocessing function for any raw text
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'\d+', '', text)
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     words = word_tokenize(text)
#     words = [lemmatizer.lemmatize(stemmer.stem(w)) for w in words if w not in stop_words]
#     return words

# # Function to process text and extract features
# def process_text_from_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         raw_text = f.read()
    
#     sentences = sent_tokenize(raw_text)
#     processed_sentences = [preprocess_text(s) for s in sentences]
#     print("Tokenized & Cleaned Sentences:")
#     print(processed_sentences)
    
#     joined_sentences = [' '.join(words) for words in processed_sentences]
    
#     # --- Bag-of-Words ---
#     bow_vectorizer = CountVectorizer()
#     X_bow = bow_vectorizer.fit_transform(joined_sentences)
#     print("\nBag-of-Words Features:")
#     print(bow_vectorizer.get_feature_names_out())
#     print(X_bow.toarray())
    
#     # --- N-grams ---
#     ngram_vectorizer = CountVectorizer(ngram_range=(1,2))
#     X_ngram = ngram_vectorizer.fit_transform(joined_sentences)
#     print("\nN-grams Features:")
#     print(ngram_vectorizer.get_feature_names_out())
    
#     # --- TF-IDF ---
#     tfidf_vectorizer = TfidfVectorizer()
#     X_tfidf = tfidf_vectorizer.fit_transform(joined_sentences)
#     print("\nTF-IDF Features:")
#     print(tfidf_vectorizer.get_feature_names_out())
#     print(X_tfidf.toarray())
    
#     # --- Word2Vec ---
#     w2v_model = Word2Vec(processed_sentences, vector_size=50, window=3, min_count=1, workers=4)
#     print("\nWord2Vec Vector for 'python':")
#     if 'python' in w2v_model.wv:
#         print(w2v_model.wv['python'])
#     else:
#         print("Word 'python' not in vocabulary.")

# # Example usage
# file_path = '/content/sample_text.txt'  # Replace with your file path
# process_text_from_file(file_path)

#---------------------------------------------------------------------------------------

#raw_text

# import re
# import string
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from gensim.models import Word2Vec

# # Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Define stopwords, stemmer, and lemmatizer
# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()

# # Preprocessing function for any raw text
# def preprocess_text(text):
#     # Lowercase
#     text = text.lower()
#     # Remove numbers
#     text = re.sub(r'\d+', '', text)
#     # Remove punctuation
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     # Tokenize into words
#     words = word_tokenize(text)
#     # Remove stopwords + stem + lemmatize
#     words = [lemmatizer.lemmatize(stemmer.stem(w)) for w in words if w not in stop_words]
#     return words

# # Function to process text and extract features
# def process_text(raw_text):
#     # Split into sentences
#     sentences = sent_tokenize(raw_text)
    
#     # Preprocess each sentence
#     processed_sentences = [preprocess_text(s) for s in sentences]
#     print("Tokenized & Cleaned Sentences:")
#     print(processed_sentences)
    
#     # Join for CountVectorizer and TF-IDF
#     joined_sentences = [' '.join(words) for words in processed_sentences]
    
#     # --- Bag-of-Words ---
#     bow_vectorizer = CountVectorizer()
#     X_bow = bow_vectorizer.fit_transform(joined_sentences)
#     print("\nBag-of-Words Features:")
#     print(bow_vectorizer.get_feature_names_out())
#     print(X_bow.toarray())
    
#     # --- N-grams ---
#     ngram_vectorizer = CountVectorizer(ngram_range=(1,2))
#     X_ngram = ngram_vectorizer.fit_transform(joined_sentences)
#     print("\nN-grams Features:")
#     print(ngram_vectorizer.get_feature_names_out())
    
#     # --- TF-IDF ---
#     tfidf_vectorizer = TfidfVectorizer()
#     X_tfidf = tfidf_vectorizer.fit_transform(joined_sentences)
#     print("\nTF-IDF Features:")
#     print(tfidf_vectorizer.get_feature_names_out())
#     print(X_tfidf.toarray())
    
#     # --- Word2Vec ---
#     w2v_model = Word2Vec(processed_sentences, vector_size=50, window=3, min_count=1, workers=4)
#     print("\nWord2Vec Vector for 'python':")
#     if 'python' in w2v_model.wv:
#         print(w2v_model.wv['python'])
#     else:
#         print("Word 'python' not in vocabulary.")

# # Example usage
# raw_text = """
# Python is GREAT!!! It's widely used for NLP, AI, and Data Science. 
# Machine Learning is amazing, and it's fun to learn. NLP can be challenging but rewarding.
# """
# process_text(raw_text)

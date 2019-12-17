import json
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from .word_embedding import *
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import operator
import math
import re 
import os
import matplotlib.pyplot as plt
import timeit
from IPython.display import Markdown, display
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class InterpretError(Exception):
    '''
    thrown when a sentence couldn't be interpreted
    '''

class Preprocessor():
    '''
    Preprocessor to reduce the tableschema dimension for a given table and question. 
    Calculates the dynamic tableschema using word embeddings, therefore doesn't always 
    produce a tableschema which contains the correct columns for any kind of question.
    '''
    def __init__(self, config: dict) -> None:
        '''
        Description:
            constructs the preprocessor object using a configuration dictionary. The given
            parameters can be looked up in the respective json file.
        Parameters:
            @config: dict = contains the configuration parameters, e.g. data directionary path
        
        '''
        self.data_dir = config['data_dir']
        
        self.language = config['language']
        self.cw = config['criteria_weights']
        self.ddic_path = config['ddic_path']
        self.ddic = pd.read_csv(os.path.join(self.data_dir, self.ddic_path), sep='\t')
        self.ddic = self._prepare_ddic(self.cw)
        self.weights = self._get_field_weights(self.cw)
        self.max_fieldnum = 20
        self.glove_words = list(set(self._flatten_list(list(self.ddic['FIELD_GLOVE'].apply(self._str2list).values))))
        
        self.glove_word2idx_path = config['glove']['word2idx_path']
        self.glove_used_wordemb_path = config['glove']['used_wordemb']
        with open(os.path.join(self.data_dir, self.glove_word2idx_path)) as file:
            self.glove_word2idx = json.load(file)
        self.reverse_glove = self._build_reverse() 
        
        self.word_emb = load_word_emb(self.data_dir, self.glove_word2idx_path, self.glove_used_wordemb_path)
        self.embedding = WordEmbeddingLite(self.word_emb)
        
        self.text2num = config['text2num']
        self.aggregation_words = config['aggregation_words']
        self.removal_words = stopwords.words(self.language) + self.aggregation_words
        
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
                
    def dynamic_tableschema(self, question: str, table: str, size: int = 8, use_cw:bool = True) -> pd.DataFrame:
        '''
        Description:
            reduces the tableschema of a given table using the question, i.e. reduces the
            tableschema to the relevant columns for the specific question.
        Parameter:
            @question: str = sentence-like string containing the question
            @table: str = technical name of the table
            @size: int, optional = number of column in the dynamic tableschema
            @use_cw: bool, optional = if True, the criteria weights from the config will 
                                      be used to modify the resultset in advance to the
                                      weights, i.e. the rating will be multiplied with the
                                      given criteria weight
        Example:
            @question = "Which companies are located in germany?"
            @table = "T001"
            -> 
            AGG	QUESTION	STATEMENT	RATING
            COUNT	How few procurement code are there?	SELECT COUNT(PCD) FROM EKKO WHERE INCO2 EQL 'N...	4.0
            NaN	For cntry for tax return de, show the date of ...	SELECT KDATE FROM EKKO WHERE LANDS EQL 'DE'	8.0
            NaN	What is the release strategy where the incote...	SELECT FRGSX FROM EKKO WHERE INCO1 EQL '21600'	0.0
            NaN	When was the payment in 60 created?	SELECT BNDDT FROM EKKO WHERE ZBD3T EQL '60'	2.0
            NaN	What release status have 10 as item number in...	SELECT FRGZU FROM EKKO WHERE PINCR EQL '10'	8.0
            COUNT	The 10 last item has how many interest calcul...	SELECT COUNT(VZSKZ) FROM EKKO WHERE LPONR EQL ...	5.0
            MAX	When the incoterms part 2 was new york what wa...	SELECT MAX(OTB_VALUE) FROM EKKO WHERE INCO2 EQ...	04. Mai
            NaN	If the purchasing doc type is nb what is the ...	SELECT EBELN FROM EKKO WHERE BSART EQL 'NB'	8.0
            NaN	What is the otb check level for the fr. bloom...	SELECT OTB_LEVEL FROM EKKO WHERE VERKF EQL 'Fr...	2.0
            SUM	If the subitem interval is 1.0, what is the su...	SELECT SUM(DPPCT) FROM EKKO WHERE UPINC EQL '1.0'	5.0

            
        '''
        question_embs = self._get_question_embeddings(question)
        if len(question_embs) == 0:
            raise InterpretError("Unable to interpret question '{}'. \
                                 Please ask your question more precise.".format(question))
        column_embs = self._get_table_embeddings(table)
        nearest_candidates = self._get_nearest_candidates(question_embs, column_embs, table, use_cw=use_cw)
        df = self._get_cols(table, nearest_candidates, max_n=size)
        return df

    def criteria_weights(self, tables: list, verbose: bool = False, figsize: tuple =(10,7)) -> None:
        '''
        Description:
            Plot the criteria weights that exist for a given table
        Parameters:
            @tables: list = list of the technical names for the tables
            @verbose: bool, optional = if set to true, points will be
                                       displayed as the names of the
                                       fields
            @figsize: tuple, optional = size of the plot
        '''
        for table in tables:
            plt.figure(figsize=figsize)
            x, y, z = [], [], []
            for key in self.weights[table]:
                fieldname = list(self.ddic['FIELD_TECHNICAL']\
                                 .loc[(self.ddic['DDIC_POSITION'] == key) & 
                                      (self.ddic['TABLE_TECHNICAL'] == table)].values)[0]
                x.append(key)
                y.append(self.weights[table][key])
                z.append(fieldname)
            x, y, z = zip(*sorted(zip(x, y, z)))
            if verbose:
                plt.scatter(x, y, label=table, alpha=0.0001, s=0.001)
                for point_x, point_y, text in zip(x,y,z):
                    if point_y < 1 and (point_x/len(x) * 100) >= 90:
                        continue
                    plt.annotate(text, (point_x, point_y))
            else:
                plt.scatter(x, y, label=table, alpha=0.5)
            plt.axhline(y=1, color='r', linestyle='-')
            if len(tables) > 1:
                plt.legend()
            plt.title('Importance of Fields')
            plt.xlabel('Position in Data Dictionary')
            plt.ylabel('Weight')
            plt.show()

    # --------------- functions for the calculation of the dynamic tableschema ---------------       
    def _get_question_embeddings(self, question: str) -> np.array:
        '''
        Description:
            transforms a question into a list of the respective glove vectors.
        Parameters:
            @question: str = the question as a string
        '''
        question_filtered = self._preprocess_question(question)
        question_lem = [self.lemmatizer.lemmatize(word) for word in question_filtered]
        question_stm = [self.stemmer.stem(word) for word in question_lem]
        question_glove = list(set([self.reverse_glove.get(word) for word in question_stm]))
        embedding_words = [word for word in question_glove if word]
        embeddings = [self.embedding(word) for word in embedding_words]
        return np.array(embeddings)
    
    def _get_table_embeddings(self, table: str) -> np.array:
        '''
        Description:
            transform the columns into a list of the respective glove vectors.
        Parameters:
            @table: str = technical name of the table
        '''
        columns = self.ddic['FIELD_GLOVE_LONG'].loc[self.ddic['TABLE_TECHNICAL'] == table].values
        embeddings = []
        for column in columns:
            embs = []
            try:
                column = [col.strip() for col in column\
                          .replace('[', '')\
                          .replace(']', '')\
                          .replace("'", '')\
                          .split(',')]
            except:
                pass
            for word in column:
                embs.append(self.embedding(word))
            embs = np.array(embs)
            embeddings.append(embs)
        return np.array(embeddings)
    
    def _get_nearest_candidates(self, question_embs: np.array, column_embs: np.array, table: str, delta: float = 0.000001, use_cw: bool = True) -> list:
        '''
        Description:
            get the column candidates which are "nearest" to the content of the question given the table.
        Parameters:
            @question_embs: np.array = 
            @column_embs: np.array = 
            @table: str = 
            @delta: float, optional = 
            @use_cw: bool, optional = 
        '''
        candidates = {}
        for cidx, cembs in enumerate(column_embs):
            cembs_angles = []
            for cemb in cembs:
                embs = [cemb]
                for qemb in question_embs:
                    embs.append(qemb)
                embs = np.array(embs) # all embeddings in one matrix
                similarities = list(map(lambda x: self._clip(x), cosine_similarity(embs)[0][1:])) # similarity to the question word, vals are being clipped to avoid rounding errors (-> 1.00000000001 ==> 1)
                angle_in_radians = np.array([math.acos(sim) for sim in similarities])
                min_angle = np.min(angle_in_radians)
                cembs_angles.append(min_angle)
            n = len(cembs)
            candidates[cidx] = (np.sum(cembs_angles) / n) + delta
            if use_cw:
                candidates[cidx] /= self.weights[table][cidx] # as this is a minimization problem, weights will be divided instead of multiplied
        
        nearest_candidates = [(field_idx, angle) for field_idx, angle in sorted(candidates.items(), key=operator.itemgetter(1))]
        return nearest_candidates
    

    def _get_cols(self, table: str, nearest_candidates: list, max_n: int = 10) -> pd.DataFrame:
        '''
        Description:
            get the columns of the underlying data dictionary for the given nearest candidates.
        Parameters:
            @table: str = 
            @neartest_candidates: list = 
            @max_n: int, optional = 
        '''
        dyn_ddic = pd.DataFrame().reindex_like(self.ddic).dropna()
        weights = defaultdict(lambda: 1)
        while len(dyn_ddic) <= max_n:     
            
            current_fields = list(dyn_ddic['FIELD_TECHNICAL'].values)
            rows = self.ddic.loc[(self.ddic['TABLE_TECHNICAL'] == table) & (~self.ddic['FIELD_TECHNICAL'].isin(current_fields))]  
            
            for candidate_idx, (field_idx, angle) in enumerate(nearest_candidates[:10]):
                row = self.ddic.loc[(self.ddic['TABLE_TECHNICAL'] == table) & (self.ddic['DDIC_POSITION'] == field_idx)]
                field_glove = self._str2list(row['FIELD_GLOVE_LONG'].values[0])
                for word in field_glove:
                    angle *= weights[word]
                candidate = (field_idx, angle)
                nearest_candidates[candidate_idx] = candidate
                
            nearest_candidates = [(field_idx, angle) for field_idx, angle in sorted(nearest_candidates, key=operator.itemgetter(1))]
            try:
                near_idx, near_angle = nearest_candidates[0]
            except:
                if len(dyn_ddic) > 0:
                    dyn_ddic['SAP_POSITION'] = dyn_ddic['SAP_POSITION'].apply(lambda x: int(x))
                    dyn_ddic['DDIC_POSITION'] = dyn_ddic['DDIC_POSITION'].apply(lambda x: int(x))
                    dyn_ddic['FIELD_GLOVE_TOK'] = dyn_ddic['FIELD_GLOVE'].apply(self._str2list)
                    dyn_ddic['FIELD_GLOVE'] = dyn_ddic['FIELD_GLOVE_TOK'].apply(lambda x: ' '.join(x))
                    dyn_ddic['FIELD_GLOVE_LONG_TOK'] = dyn_ddic['FIELD_GLOVE_LONG'].apply(self._str2list)
                    dyn_ddic['FIELD_GLOVE_LONG'] = dyn_ddic['FIELD_GLOVE_LONG_TOK'].apply(lambda x: ' '.join(x))
                    return dyn_ddic[['FIELD_TECHNICAL', 'FIELD_DESCRIPTIVE', 'FIELD_GLOVE', 'FIELD_GLOVE_TOK', 'FIELD_GLOVE_LONG',\
                                     'FIELD_GLOVE_LONG_TOK', 'DDIC_POSITION', 'SAP_POSITION']].reset_index(drop=True)
                else:
                    print(table)
            row = self.ddic.loc[(self.ddic['DDIC_POSITION'] == near_idx) & (self.ddic['TABLE_TECHNICAL'] == table)]
            dyn_ddic = pd.concat([dyn_ddic, row])
            
            for words in row['FIELD_GLOVE_LONG']:
                words = self._str2list(words)
                for word in words:
                    weights[word] *= 1.05
            
            nearest_candidates = nearest_candidates[1:]
            
        dyn_ddic['SAP_POSITION'] = dyn_ddic['SAP_POSITION'].apply(lambda x: int(x))
        dyn_ddic['DDIC_POSITION'] = dyn_ddic['DDIC_POSITION'].apply(lambda x: int(x))
        dyn_ddic['FIELD_GLOVE_TOK'] = dyn_ddic['FIELD_GLOVE'].apply(self._str2list)
        dyn_ddic['FIELD_GLOVE'] = dyn_ddic['FIELD_GLOVE_TOK'].apply(lambda x: ' '.join(x))
        dyn_ddic['FIELD_GLOVE_LONG_TOK'] = dyn_ddic['FIELD_GLOVE_LONG'].apply(self._str2list)
        dyn_ddic['FIELD_GLOVE_LONG'] = dyn_ddic['FIELD_GLOVE_LONG_TOK'].apply(lambda x: ' '.join(x))
        return dyn_ddic[['FIELD_TECHNICAL', 'FIELD_DESCRIPTIVE', 'FIELD_GLOVE', 'FIELD_GLOVE_TOK', 'FIELD_GLOVE_LONG',\
                         'FIELD_GLOVE_LONG_TOK', 'DDIC_POSITION', 'SAP_POSITION']].reset_index(drop=True)    
    # --------------- initialization functions ---------------
    def _prepare_ddic(self, cw: dict, top_n: int = 20) -> pd.DataFrame:
        '''
        Description:
            prepares the data dictionary by preparing the columns for the criteria weights
        Parameters:
            @cw: dict = 
            @top_n: int, optional = 
        '''
        def check_pos(row, n):
            if row['DDIC_POSITION'] <= n:
                return 'X'
            else:
                return ''
    
        def check_fit(row, criteria):
            fits = False
            for crit in criteria:
                if row[crit] == 'X':
                    fits = True
            if fits:
                return ''
            else:
                return 'X'
        criteria = cw.keys()
        self.ddic['TOP_N'] = ''
        self.ddic['OTHER'] = ''
        self.ddic['TOP_N'] = self.ddic.apply(lambda row: check_pos(row, n=top_n), axis=1)
        self.ddic['OTHER'] = self.ddic.apply(lambda row: check_fit(row, criteria), axis=1) 
        return self.ddic

    def _get_field_weights(self, cw: dict) -> dict:
        '''
        Description:
            get the weights for each field characteristic (e.g. fields that are primary keys get a weight of 1.2)
        Parameters:
            @cw: dict = 
        '''
        weights = {}
        for table in self.ddic['TABLE_TECHNICAL'].unique():
            weights[table] = defaultdict(lambda: 1) # defaultdict with value 1
            for criteria, value in cw.items():
                self._change_weights(weights, table, criteria, value)
        return weights

    def _change_weights(self, weights: dict, table: str, criteria: str, val: float) -> dict:
        '''
        Description:
            helper function used by _get_field_weights to easily apply the weights for a field
        Parameters:
            @weights: dict = 
            @table: str = 
            @criteria: str = 
            @val: float = 
        '''
        fields = self.ddic['DDIC_POSITION'].loc[(self.ddic[criteria] == 'X') & (self.ddic['TABLE_TECHNICAL'] == table)].values
        for field in fields:
            weights[table][field] *= val
        return weights
    
    def _build_reverse(self) -> dict:
        '''
        Description:
            Builds a dictionary which maps the stemmed version of a 
            word to the appearance of the word in the glove file
        '''
        reverse_glove = {}
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        for row in self.glove_word2idx:
            processed_row = stemmer.stem(lemmatizer.lemmatize(row))
            try:
                reverse_glove[processed_row]
            except:
                reverse_glove[processed_row] = row
        return reverse_glove
    
    # --------------- question processing functions, get used only by self._get_question_embeddings() ---------------
    
    def _split_camel_case(self, toks_cc: list) -> list:
        '''
        Description:
            Splits a list of word tokens when camel case is being found
        Parameters:
            @toks_cc: list = list of word tokens
        Example:
            @toks_cc: ['This', 'isAn', 'example'] 
            -> ['This', 'is', 'An', 'example']
        '''
        toks = []
        for tok in toks_cc:
            start_idx = [i for i, e in enumerate(tok) 
                     if e.isupper()] + [len(tok)] 

            start_idx = [0] + start_idx 
            toks.append([tok[x: y] for x, y in zip(start_idx, start_idx[1:])])
        toks = [item for sublist in toks for item in sublist]
        toks = [tok for tok in toks if tok != '']
        return toks
    
    def _clip(self, val: float) -> float:
        '''
        Description:
            sets an upper and lower bound for a value, i.e. values greater
            than 1 are set to 1 and values smaller than -1 are set to -1.
        Parameters:
            @val: float = value which is being clipped (if needed)
        '''
        if val > 1:
            val = 1
        elif val < -1:
            val = -1
        return val
        
    def _try_split(self, word):
        '''
        Description:
            Tries to find subwords from the given words which exist in the glove file.
        Parameters:
            @words: list = list of the strings
            @glove: dict = file containing the glove to idx mapping
        '''
        possible_cuts = []
        neo_words = []
        chars = list(word)
        substr = ''.join(chars[:4])
        for char in chars[4:]:
            substr += char
            if substr in self.glove_word2idx:
                possible_cuts.append(len(substr))
        for possible_cut in possible_cuts:
            word[possible_cut:]
            if word[possible_cut:] in self.glove_word2idx and len(word[possible_cut:]) > 3:
                neo_words.append([word[:possible_cut], word[possible_cut:]])
        flat_neo_words = [item for sublist in neo_words for item in sublist]
        return flat_neo_words
    
    def _glove_substr(self, words: list) -> list:
        '''
        Description:
            finds substrings from the list of words which appear in the glove file
        Parameters:
            @words: list = list of words which are being checked
        '''
        glove_substrs = []
        for word in words:
            if word not in self.glove_word2idx:
                chars = list(word)
                glove_sub = ''.join(chars[:2])
                for char in chars[2:]:
                    glove_sub += char
                    if glove_sub in self.glove_word2idx:
                        glove_substrs.append(glove_sub)
                        break
        return glove_substrs
    
    def _rmv_num_words(self, token: list) -> list:
        '''
        Description:
            Filters numerical values out of a list of word token
        Parameters:
            @token: list containing the word token
        Example:
            @token = ['This', 'is', '1', 'example']
            -> ['This', 'is', 'example']
        '''
        num_words = [(idx, int(word)) for idx, word in enumerate(token) if word.isdigit()]
        for idx, num_word in num_words:
            if num_word > self.max_fieldnum or self._is_decimal(num_word):
                if str(num_word) in token:
                    token.remove(str(num_word))
        return token
    
    def _is_decimal(self, number: float) -> bool:
        '''
        Description:
            checks if a number is a strict decimal, i.e. not an int but a float type.
            Separator used for the check is a point, e.g. '42.0', which would be seen
            as a decimal.
        Parameters:
            @number: float = number which needs to be checked
        Example:
            @number = '1' -> False
            ---
            @number = '1.0' -> True
            ---
            @number = '.0' -> True
            ---
            @number = '1.' -> True
        '''
        number = str(number)
        if '.' in number:
            if number.startswith('.'):
                prf = 0
                try:
                    suf = number.split('.')[1]
                except UnboundLocalError:
                    return False
            elif number.endswith('.'):
                suf = 0
                try:
                    prf = number.split('.')[0]
                except UnboundLocalError:
                    return False
            else:
                try:
                    prf = number.split('.')[0]
                    suf = number.split('.')[1]
                except UnboundLocalError:
                    return False
        else:
            return False
        try:
            int(prf)
            int(suf)
            return True
        except ValueError:
            return False
        
    def _flatten_list(self, list2d: list) -> list:
        '''
        Description:
            flattens a list, i.e. removes additional dimensions
        Parameters:
            @list2d: list = list with content in multiple dimensions
        Example:
            @list2d = [['Flatten', 'this'], 'list']
            -> ['Flatten', 'this', 'list']
        '''
        return [item for sublist in list2d for item in sublist]
    
    def _str2list(self, words: str) -> list:
        '''
        Description:
            turns list-like strings into lists
        Parameters:
            @words: str = list that has been serialized as a string,
                          i.e. the same as str(list([...]))
        Example:
            @words: "['company', 'code']"
            -> ['company', 'code']
        '''
        if type(words) != str:
            return words
        else:
            rem_chars = "['\"]"
            if '[' in words \
            and ']' in words:
                for char in rem_chars:
                    words = words.replace(char, '')
                words_tok = [word.strip() for word in words.split(',')]
                return words_tok
            else:
                return words
    
    def _preprocess_question(self, question: str) -> list:
        '''
        Description:
            preprocesses a question by applying the following techniques to it:
            1.  transforming the question to lowercase
            2.  remove decimal words from the question
            3.  replace punctuation
            4.  transform ordinal numbers to regular numbers 
                (e.g. 'first' to '1', or '1st' to '1')
            5.  split at camel case positions if given
            6.  search for neologisms in the question 
                (e.g. 'salesperson' as 'sales' and 'person')
            7.  search for substrings in the question
                (e.g. 'scmprocess' includes 'scm', which appears in the glove file)
            8.  remove removal words (i.e. stopwords and additional words provided
                in the configuration file)
            9.  remove numerical words that are unlikely to be related to column names
                (as sql like values are not being needed by the algorithm)
        Parameters:
            @question: str = string containing the question
        '''
        question = question.lower()
        question_tok = [word for word in word_tokenize(question) if not self._is_decimal(word)]
        for idx, tok in enumerate(question_tok):
            for punct in string.punctuation:
                tok = tok.replace(punct, ' ')
            if self.text2num.get(tok):
                question_tok[idx] = self.text2num.get(tok)
            else:
                question_tok[idx] = tok
        question_ncc = [word for word in self._split_camel_case(question_tok)]
        question_neo = self._flatten_list([self._try_split(word) for word in question_ncc])
        question_sub = self._glove_substr(question_ncc)
        question_ncc += question_neo + question_sub
        question_rmv = [word.lower() for word in question_ncc if word not in self.removal_words]
        question_filtered = [tok for tok in self._rmv_num_words(question_rmv) if  not tok.isspace()]
        return question_filtered  
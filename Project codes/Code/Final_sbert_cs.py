import pickle
from nltk.corpus import stopwords
import re
import nltk
from nltk.stem import PorterStemmer
ps=PorterStemmer()
import math
import numpy as np
from numpy.linalg import norm
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


def sbert_sum(query):
   

    #importing the objects as pickle files
    with open('Code/posting_lst.pickle','rb') as file1:
        term_f=pickle.load(file1)

    with open('Code/idf_lst.pickle','rb') as file2:
        idf_dict=pickle.load(file2)

    with open('Code/files.pickle','rb') as file3:
        file_lst=pickle.load(file3)

    with open('Code/Doc_len.pickle','rb') as file4:
        doc_len=pickle.load(file4)

    with open('Code/Glove_100d_dict.pickle','rb') as file5:
        glove_dict=pickle.load(file5)

    with open('Code/Replace_dictionary.pickle','rb') as file6:
        new_replace_dict=pickle.load(file6)

    #creating a set of stopwords in english
    stpwrds = set(stopwords.words('english'))

    #function used to preprocess the given input query into tokens 
    def query_preprocess(query1):
        #removing the characters except alphanumeric and underscore
        tempq = re.compile('[^\w\s]')
        query1 = re.sub(tempq,' ',query1)
        #removing non-ascii characters
        encoded_string = query1.encode("ascii", "ignore")
        query2 = encoded_string.decode()
        #tokenization
        tokens=nltk.word_tokenize(query2)
        tokens_lwr=[]
        #converting tokens to lower case
        for i in tokens:
            tokens_lwr.append(i.lower())
        tokens_nmbr=[]
        #numbers are converted to their respective word format using inbuilt function 
        for z in tokens_lwr:
            try:
                z=num2words(int(z))
            except:
                pass
            tokens_nmbr.append(z)
        tkns=[]
        #stemming the tokens
        for p in tokens_nmbr:
            tkns.append(ps.stem(p))
        tkns1=[]
        #removing stopwords
        for p in tkns:
            if p not in stpwrds:
                tkns1.append(p)
        tkns_1=[]
        #removing tokens not present in all of the documents at all
        for p in tkns1:
            if p in idf_dict:
                tkns_1.append(p)
        return tkns_1

    count=0
    N=len(file_lst)
    for i in range(len(doc_len)):
        count=count+doc_len[i]
    avg_len=count/N
    avg_len

    def idf(p):
        #idf function calculated for help in finding similarity score 
        doc_f=0
        if p in idf_dict:
            doc_f1=(len(file_lst)/idf_dict[p])
        idf_val2=math.log((N-doc_f1+0.5)/(doc_f1+0.5)+1)
        return idf_val2

    score={}
    query_pp=query_preprocess(query)
    k=1.2
    b=0.75
    for i in range(len(file_lst)):
            score[i]=0
            for j in query_pp:
                tm_f=0
                if j in term_f:
                    if i in term_f[j]:
                        tm_f=term_f[j][i]
                idf_val=idf(j)
                #calculating similarity score for document and query token words and storing in dictionary
                scr_i=idf_val*(k+1)*tm_f/(tm_f+k*(1-b+b*(doc_len[i]/avg_len)))
                score[i]+=scr_i


    score1=sorted(score.items(),key=lambda x: x[1],reverse=True)

    #after sorting the dictionary of similarity score in descending order and printing the top 10 documents
    new_file_lst=[]
    count = 20
    for i in score1:
        if(i[1]>0):
            new_file_lst.append(file_lst[i[0]])

    def clean_wrd(word):
        tempq = re.compile('[^0-9a-zA-Z]')
        word_lst = re.split(tempq,word)
        if len(word_lst) <= 1:
            return word_lst[0]
        elif len(word_lst[0])>len(word_lst[1]): 
            return word_lst[0]
        else:
            return word_lst[1]

    file=open(r'New_Ramayana_Dataset/'+new_file_lst[0])
    file_txt_raw=file.read()
    file_txt_raw=file_txt_raw.replace('\n',' ')

    string=""
    quote=False
    delimiters=[",",".","?","!"]
    del_map={"commark":",","dot":".","qmark":"?","exmark":"!"}
    rev_del_map={v:k for k,v in del_map.items()}
    c=0
    while c<len(file_txt_raw):
        if file_txt_raw[c]=='"':
            string=string+'"'
            c=c+1
            while c<len(file_txt_raw) and file_txt_raw[c]!='"':
                if file_txt_raw[c] not in delimiters:
                    string=string+file_txt_raw[c]
                    c=c+1
                else:
                    string=string+" "+rev_del_map[file_txt_raw[c]]
                    c=c+1
            string=string+'".'
            c=c+1
        if c>=len(file_txt_raw):
            break
        string=string+file_txt_raw[c]
        c=c+1



    file_txt=sent_tokenize(string)

    new_file_txt=[]
    sent_c=0
    word_c=0
    for i in file_txt:
        j=i.replace('commark',',')
        k=j.replace('dot','.')
        l=k.replace('qmark','?')
        m=l.replace('exmark','!')
        new_file_txt.append(m)



    count1=0
    sen_li=[]
    sent_word_vec=[]
    for i in new_file_txt:
        all_wrd_vec=[]
        wrd_lst=i.split(' ')
        new_wrd_lst=[]
        for p in wrd_lst:
            new_wrd_lst.append(p.lower())
        new_wrd_lst1=[]
        for m in new_wrd_lst:
            new_wrd_lst1.append(clean_wrd(m))
        new_wrd_lst2=[]
        for j1 in new_wrd_lst1:
            if(len(j1)>1):
                if(j1[-2]=='\''):
                    j1=j1[:-2]
            if j1 in new_replace_dict:
                j1=new_replace_dict[j1]
            new_wrd_lst2.append(j1)
        s=" ".join(new_wrd_lst2)
        sen_li.append(s)





    query_vect=[]
    avg_vec=[]
    for i in [query]:
        all_wrd_vec=[]
        wrd_lst=i.split(' ')
        new_wrd_lst=[]
        for p in wrd_lst:
            new_wrd_lst.append(p.lower())
        new_wrd_lst1=[]
        for m in new_wrd_lst:
            new_wrd_lst1.append(clean_wrd(m))
        new_wrd_lst2=[]
        for j1 in new_wrd_lst1:
            if(len(j1)>1):
                if(j1[-2]=='\''):
                    j1=j1[:-2]
            if j1 in new_replace_dict:
                j1=new_replace_dict[j1]
            new_wrd_lst2.append(j1)
        s=" ".join(new_wrd_lst2)
        query_vect.append(s)


    query_vect

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sen_li)
    embeddings_query = model.encode(query_vect)

    for i in range(len(embeddings)):
        embeddings[i]=np.array(embeddings[i])
    for j in range(len(embeddings_query)):
        embeddings_query[j]=np.array(embeddings_query[j])

    embeddings_query[0].shape

    new_score={}
    for i in range(len(sen_li)):
        new_score[i]=float(cosine_similarity(embeddings[i].reshape(1,384),embeddings_query.reshape(1,384)))



    new_score=dict(sorted(new_score.items(), key=lambda item: item[1],reverse=True))



    query_sents={}
    c=0
    for i in new_score:
        if c==10:
            break
        query_sents[i]=(new_file_txt[i])
        c+=1

    sim_mat = np.zeros([len(query_sents), len(query_sents)])

    row=0
    for i in query_sents:
        col=0
        for j in query_sents:
            if i != j:

                sim_mat[row][col] = cosine_similarity(embeddings[i].reshape(1,384), embeddings[j].reshape(1,384))[0,0]
            col+=1
        row+=1




    import networkx as nx

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(query_sents.values())),reverse=True)[:5]

    ind_lst=[]
    for snt in query_sents:
        for pp in range(len(ranked_sentences)):
            if query_sents[snt] == ranked_sentences[pp][1]:
                ind_lst.append(snt)
    ind_lst.sort()
    summary1=''
    sn = 5
    for idx in ind_lst:
        summary1=summary1+(query_sents[idx])
        summary1+=('\n')

    

    eu_mat = np.zeros([len(query_sents), len(query_sents)])
    row=0
    for i in query_sents:
        col=0
        x=pd.Series(embeddings[i])
        for j in query_sents:
            if i != j:
                y=pd.Series(embeddings[j])
                eu_mat[row][col] = np.linalg.norm(x-y)
            col+=1
        row+=1




    import networkx as nx

    nx_graph = nx.from_numpy_array(eu_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(query_sents.values())))[:5]

    ind_lst=[]
    for snt in query_sents:
        for pp in range(len(ranked_sentences)):
            if query_sents[snt] == ranked_sentences[pp][1]:
                ind_lst.append(snt)
    ind_lst.sort()
    summary2=''
    sn = 4
    for idx in ind_lst:
        summary2=summary2+(query_sents[idx])
        summary2+=('\n')

    return summary1,summary2






# In[ ]:





# In[ ]:





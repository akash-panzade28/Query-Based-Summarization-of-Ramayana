from flask import Flask, render_template, request
from nltk.corpus import stopwords

import Code.Final_Glove as glb
import Code.Final_CS_fasttext as ft
import Code.Final_sbert_cs as sbrt
import Code.Final_clusters as clust
import random
import torch
import pickle
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cpu'

tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

#importing the objects as pickle files
with open('Code/posting_lst.pickle','rb') as file1:
    term_f=pickle.load(file1)

with open('Code/idf_lst.pickle','rb') as file2:
    idf_dict=pickle.load(file2)

with open('Code/files.pickle','rb') as file3:
    file_lst=pickle.load(file3)

with open('Code/Doc_len.pickle','rb') as file4:
    doc_len=pickle.load(file4)

with open('Code/fasttext_100d_dict.pickle','rb') as file5:
    fst=pickle.load(file5)

with open('Code/Replace_dictionary.pickle','rb') as file6:
    new_replace_dict=pickle.load(file6)

#creating a set of stopwords in english
stpwrds = set(stopwords.words('english'))



app = Flask(__name__)


@app.route('/', methods = ['GET','POST'])
def hello_world():

    # for checkbox
    # if request.form.get('option1'):
    #     print("opetion1")

    # if request.form.get('option2'):
    #     print("option2")

    # for file upload
    # if request.method == 'POST':
    #   f = request.files['apk']
    #   filename = secure_filename(f.filename)
    #   f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # return render_template('index.html')
    return render_template('index.html')

@app.route('/uploader', methods = ['GET','POST'])
def uploader():
    # return render_template('index.html')
    if request.method == "POST":
       # getting input with name = fname in HTML form
        query = request.form.get("query")
        fl = open("static/query.txt",'w')
        fl.write(query)
        fl.close()
    #    try:
    #         ans = glb.extr_summ_cs_glov(query)
    #    except:
    #        ans = "Invalid Query!!!!"
           
    return render_template('options.html',query = query)

@app.route('/glove', methods = ['GET','POST'])
def glove():
    
    fl = open("static/query.txt","r")
    query = fl.read()
    fl.close()

    # try:   
    ans1,ans2 = glb.glove_summ(query)
    return render_template('glove.html',ans1 = ans1,ans2=ans2)

    # except:
    #     return render_template("index.html",err = "Invalid Query")


@app.route('/fasttext', methods = ['GET','POST'])
def fasttext():
    
    fl = open("static/query.txt","r")
    query = fl.read()
    fl.close()

     
    ans1,ans2 = ft.fasttext_sum(query,model,tokenizer,term_f,idf_dict,file_lst,doc_len,fst,new_replace_dict,stpwrds,torch_device,model_name)
    return render_template('fasttext.html',ans1 = ans1,ans2=ans2)

    # except:
    #     return render_template("index.html",err = "Invalid Query")

@app.route('/sbert', methods = ['GET','POST'])
def sbert():
    
    fl = open("static/query.txt","r")
    query = fl.read()
    fl.close()
   
    ans1,ans2 = sbrt.sbert_sum(query)
    return render_template('sbert.html',ans1 = ans1,ans2=ans2)

    # except:
    #     return render_template("index.html",err = "Invalid Query")   

@app.route('/cluster', methods = ['GET','POST'])
def cluster():
    
    fl = open("static/query.txt","r")
    query = fl.read()
    fl.close()

    try:   
        ans1 = clust.cluster_sum(query)
        return render_template('cluster.html',ans1 = ans1)

    except:
        return render_template("index.html",err = "Invalid Query") 

@app.route('/characters')
def characters():
    
    # fl = open("static/characters.txt","r")
    # query = fl.read()
    # query = query.split("\n")
    # fl.close()

    
    return render_template('characters.html')

    


if __name__ == "__main__":
    app.run(debug=True)
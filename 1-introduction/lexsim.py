# -*- coding: utf-8 -*-
# PARA PYTHON 2.7
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import *
from nltk.stem.porter import *
from nltk.corpus import wordnet as wn

import math
import numpy as np
from struct import *


from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

stemmer = PorterStemmer()

# funcion que lee un dataset de similiud lexica
def leer_dataset(nombre_archivo):
    dataset={}
    source_file=open(nombre_archivo,"r")
    for linea in source_file.readlines():
        posicion_primer_tab=linea.find("\t")
        posicion_segundo_tab=linea.find("\t",posicion_primer_tab+1)


        palabra1=linea[0:posicion_primer_tab]
        palabra2=linea[posicion_primer_tab+1:posicion_segundo_tab]
        gold_standard=linea[posicion_segundo_tab+1:-1]
        gold_standard=float(gold_standard)

        dataset[(palabra1,palabra2)]=gold_standard
    return dataset


# funci�n que representa una lista como una lista de n-gramas
def n_grams(lista,n):
    ngrams=[]
    for i in range(len(lista)-n+1):
        ngrams.append(lista[i:i+n])
    return ngrams

def n_grams_spectra(lista,n_desde,n_hasta): # ojo cuidar que n_desde<n_hasta
    ngrams=[]
    for n in range(n_desde,n_hasta+1):
        ngrams+=n_grams(lista,n)
    return ngrams

#prueba de ngramas
#print n_grams(["El","perro","mordi�","al","ni�o"],2)

#print n_grams_spectra("murcielago",2,3)
#exit()

####################################################################
# FUNCIONES DE SIMILITUD LEXICA MORFOLOGICA
####################################################################
def lex_sim_jaccard(palabra1,palabra2):
    A=set(palabra1)
    B=set(palabra2)
    AuB=A.union(B)
    AiB=A.intersection(B)
    similitud=float(len(AiB))/float(len(AuB))
    return similitud

def lex_sim_jaccard_ngrams(palabra1,palabra2,n_desde=2,n_hasta=3):
    A=n_grams_spectra(palabra1,n_desde,n_hasta)
    B=n_grams_spectra(palabra2,n_desde,n_hasta)
    return lex_sim_jaccard(A,B)


def lex_sim_cosine(palabra1,palabra2):
    A=set(palabra1)
    B=set(palabra2)
    AuB=A.union(B)
    AiB=A.intersection(B)
    similitud=float(len(AiB))/((len(A)*len(B))**0.5)
    return similitud

def lex_sim_sorensen(palabra1,palabra2):
    A=set(palabra1)
    B=set(palabra2)
    AiB=A.intersection(B)
    similitud=float(len(AiB))/(len(A)+len(B))
    return similitud

def lex_sim_sorensen_ngrams(palabra1,palabra2,n_desde=2,n_hasta=3):
    stem1 = stemmer.stem(palabra1)
    stem2 = stemmer.stem(palabra2)
    A = n_grams_spectra(stem1,n_desde,n_hasta)
    B = n_grams_spectra(stem2,n_desde,n_hasta)
    try:
        ans = lex_sim_sorensen(A,B)
    except:
        ans = 0
    return ans

def lex_sim_Jaro(word1,word2):

    # determines the longer and shorter strings
    if len(word1)>len(word2):
        long_word=word1
        short_word=word2
    else:
        long_word=word2
        short_word=word1
    # establish the window size
    window=len(long_word)/2-1
    if window<0:
        window=0
    # initializes list of matching positions
    long_matches=[]
    short_matches=[]
    tlong_word=long_word
    for i in range(0,len(short_word)):
        lower_bound=max([0,i-window])
        upper_bound=min([len(tlong_word)-1,i+window])
        j=tlong_word.find(short_word[i],lower_bound,upper_bound+1)
        if j>=0: # sucsessfull match
            tlong_word=tlong_word[0:j]+'#'+tlong_word[j+1:]  # the character can't be matched again!
            long_matches.append(j)
            short_matches.append(i)
    matches=len(long_matches)  # or len(short_matches
    long_matches.sort()
    transpositions=0
    for i in range(0,matches):
        if not(short_word[short_matches[i]]==long_word[long_matches[i]]):
            transpositions=transpositions+1
    transpositions=1.0*transpositions/2
    if matches==0:
            return 0.0
    return ((1.0*matches/len(short_word))+(1.0*matches/len(long_word))+((0.0+matches-transpositions)/matches))/3.0



# Edit distance
#
# Levenshtein VI (1966). "Binary codes capable of correcting deletions, insertions, and reversals". Soviet Physics Doklady 10: 707�10.
# (http://en.wikipedia.org/wiki/Edit_distance)
# R.A. Wagner and M.J. Fischer. 1974. The String-to-String Correction Problem. Journal of the ACM, 21(1):168�173.
def edit_distance(word1,word2):
    d0=range(0,len(word2)+1)
    d1=range(0,len(word2)+1)
    for i in range(1,len(word1)+1):
        d1[0]=i
        for j in range(1,len(word2)+1):
            deletion_cost=1
            insertion_cost=1
            if word1[i-1]==word2[j-1]:
                substitution_cost=0
            else:
                 substitution_cost=1
            d1[j]=min([d1[j-1]+insertion_cost,d0[j]+deletion_cost,d0[j-1]+substitution_cost])
        for k in range(0,len(word2)+1):
            d0[k]=d1[k]
    max_distance=max([len(word1),len(word2)])
    min_distance=0
    return d1[len(word2)]

def lex_sim_edit_distance(word1,word2):
    return 1-float(edit_distance(word1,word2))/max([len(word1),len(word2)])
####################################################################







####################################################################
# FUNCIONES DE SIMILITUD BASADAS EN CONOCIMIENTO (WORDNET)
####################################################################

# similitud basada en contar el m�nimo numero de arcos entre pares de posibles conceptos(synsets)
def lex_sim_path(lemma1,lemma2):
    try:
        synsets_lemma1=wn.synsets(lemma1)
        synsets_lemma2=wn.synsets(lemma2)
    except:
        return 0.0
    max_sim=0
    for synset1 in synsets_lemma1:
        for synset2 in synsets_lemma2:
            sim=synset1.path_similarity(synset2)
            if sim>max_sim:
                max_sim=sim
    return max_sim

# igual que "path" pero usando la similitud de Leacock-Chodorow
def lex_sim_lch(lemma1,lemma2):
    try:
        synsets_lemma1=wn.synsets(lemma1)
        synsets_lemma2=wn.synsets(lemma2)
    except:
        return 0.0
    max_sim=0
    for synset1 in synsets_lemma1:
        for synset2 in synsets_lemma2:
            try:
                sim=synset1.lch_similarity(synset2)
            except:
                sim=0
            if sim>max_sim:
                max_sim=sim
    return max_sim


# Similitud de Wu-Palmer
def lex_sim_wup(lemma1,lemma2):
    try:
        synsets_lemma1=wn.synsets(lemma1)
        synsets_lemma2=wn.synsets(lemma2)
    except:
        return 0.0

    max_sim=0
    for synset1 in synsets_lemma1:
        for synset2 in synsets_lemma2:
            try:
                sim=synset1.wup_similarity(synset2)
            except:
                sim=0
            if sim>max_sim:
                max_sim=sim
    return max_sim


# Similitud de Paul Resnik
def lex_sim_res(lemma1,lemma2,information_content=brown_ic):
    try:
        synsets_lemma1=wn.synsets(lemma1)
        synsets_lemma2=wn.synsets(lemma2)
    except:
        return 0.0
    max_sim=0
    for synset1 in synsets_lemma1:
        for synset2 in synsets_lemma2:
            try:
                sim=synset1.res_similarity(synset2,information_content)
            except:
                sim=0
            if sim>max_sim:
                max_sim=sim
    return max_sim

# Similitud de Jiang-Conrath
def lex_sim_jcn(lemma1,lemma2,information_content=brown_ic):
    try:
        synsets_lemma1=wn.synsets(lemma1)
        synsets_lemma2=wn.synsets(lemma2)
    except:
        return 0.0
    max_sim=0
    for synset1 in synsets_lemma1:
        for synset2 in synsets_lemma2:
            try:
                sim=synset1.jcn_similarity(synset2,information_content)
            except:
                sim=0
            if sim>max_sim:
                max_sim=sim
    return max_sim


# Similitud de Dekang Lin
def lex_sim_lin(lemma1,lemma2,information_content=brown_ic):
    try:
        synsets_lemma1=wn.synsets(lemma1)
        synsets_lemma2=wn.synsets(lemma2)
    except:
        return 0.0

    max_sim=0
    for synset1 in synsets_lemma1:
        for synset2 in synsets_lemma2:
            try:
                sim=synset1.lin_similarity(synset2,information_content)
            except:
                sim=0
            if sim>max_sim:
                max_sim=sim
    return max_sim







####################################################################
# FUNCIONES DE SIMILITUD CON WORD EMBEDDINGS
####################################################################

# GET WORD2VEC EMBEDDINGS
# iterates over the pretrained word2vec vectors
word2vec={}
def init_w2v(word2vec):
    input_file=open("./word2vec/GoogleNews-vectors-negative300.bin","rb")
    number_of_words,dimmensions=map(int,input_file.readline().split())
    record_size=4*dimmensions
    structure="f"*dimmensions
    print "word2vec dimmensions={1}, vocabulary size={0}".format(number_of_words,dimmensions)
    i=0
    print "looking for embeddings of target vocabulary in word2vec"

    for i in range(number_of_words):
        word=[]
        while True:
            ch=input_file.read(1)
            if ch == b' ':
                break
            if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                word.append(ch)
        word=b''.join(word)
        word=word.decode("utf-8",errors="ignore")#.lower()
        vector=unpack(structure,input_file.read(record_size))

        if not word in word2vec:
            #print "\t",count_found_words,word
            #if word2vec[word]==None:
            word2vec[word]=np.array(vector)

        if i % 100000==0:
            print "{0}% words processed".format(i / 20000)
        if i > 2000000:   #OJO AQUI SE LIMITA EL N�MERO DE PALABRAS A USAR DE WORD2VEC
            print ''
            break
    return word2vec



def lex_sim_word2vec(palabra1,palabra2):
    if len(word2vec)==0:
        init_w2v(word2vec)
    try:
        sim = 1 - correlation(word2vec[palabra1],word2vec[palabra2])
    except:
        return 0
    if sim<=0:
        sim=0
    return sim
####################################################################

##########################################
##### Resultados con 500000 muestras #####
##########################################
#     cosine             0.5946          #
#     correlation        0.5945          #
#     braycurtis         0.5851          #
#     russellrao         0.5508          #
#     kulsinski          0.5457          #
#     chebyshev          0.3771          #
#     hamming            0.1557          #
#     jaccard            0.1557          #
#     rogerstanimoto     0.1352          #
#     sokalmichener      0.1352          #
#     matching           0.0855          #
#     dice               0.0336          #
#     sokalsneath       -0.0153          #
#     yule              -0.0222          #
##########################################



####################################################################
# FUNCIONES DE SIMILITUD HIBRIDAS
####################################################################
def lex_sim_path_edit_distance(palabra1,palabra2):
    sim=lex_sim_path(palabra1,palabra2)
    if sim==0:
        sim=lex_sim_edit_distance(palabra1,palabra2)
    return sim

def lex_sim_path_jaccard_23grams_porter(palabra1,palabras2):
    sim=lex_sim_path(palabra1,palabra2)
    if sim==0:
        _palabra1=stemmer.stem(palabra1)
        _palabra2=stemmer.stem(palabra2)
        sim=lex_sim_jaccard_ngrams(_palabra1,_palabra2,2,3)
    return sim










if __name__ == '__main__':  # ESTE "IF" ES PARA QUE LA SIGUIENTE PARTE DEL CODIGO NO SE EJECUTE CUANDO ESTE PROGRAMA SE IMPORTE EN OTRO PROGRAMA CON import lexsim
    nombres_datasets=["MC","MEN","MTURK287","MTURK771","REL122","RG","RW","SCWS","SL999","VERB143","WS353","WSR","WSS","YP130"]
    suma_Pearson_r=0.0
    suma_numero_de_pares=0
    print "Dataset\t#pares\t(Pearson_r,p-value)"
    for nombre_dataset in nombres_datasets:
        dataset=leer_dataset("./data_lexsim/"+nombre_dataset+".txt")
        predicciones=[]
        gold_standard=[]
        for palabra1,palabra2 in dataset:
            stem1=stemmer.stem(palabra1)
            stem2=stemmer.stem(palabra2)


            # QUITAR EL # A LA LINEA DE LA FUNCION DE SIMILITUD LEXICA A USAR
            #prediccion=lex_sim_cosine(stem1,stem2)
            #prediccion=lex_sim_cosine(palabra1,palabra2)
            #prediccion=lex_sim_jaccard(stem1,stem2)
            #prediccion=lex_sim_jaccard(palabra1,palabra2)
            #prediccion=lex_sim_jaccard_ngrams(stem1,stem2)  #** 0.1989
            #prediccion=lex_sim_jaccard_ngrams(palabra1,palabra2)
            #prediccion=lex_sim_Jaro(stem1,stem2)
            #prediccion=lex_sim_Jaro(palabra1,palabra2)
            #prediccion=lex_sim_edit_distance(stem1,stem2)
            #prediccion=lex_sim_edit_distance(palabra1,palabra2)
            # EN LAS SIGUIENTES YA NO TIENE SENTIDO USAR STEMS
            #prediccion=lex_sim_path(palabra1,palabra2)
            #prediccion=lex_sim_lch(palabra1,palabra2)
            #prediccion=lex_sim_wup(palabra1,palabra2)
            #prediccion=lex_sim_res(palabra1,palabra2)
            #prediccion=lex_sim_jcn(palabra1,palabra2)
            #prediccion=lex_sim_jcn(palabra1,palabra2,information_content=semcor_ic)
            #prediccion=lex_sim_lin(palabra1,palabra2)
            #prediccion=lex_sim_lin(palabra1,palabra2,information_content=semcor_ic)
            #prediccion=lex_sim_path_edit_distance(palabra1,palabra2)
            #prediccion=lex_sim_path_jaccard_23grams_porter(palabra1,palabra2)
            prediccion=lex_sim_word2vec(palabra1,palabra2)

            #prediccion=lex_sim_sorensen_ngrams(palabra1,palabra2)

            predicciones+=[prediccion]
            GS=dataset[(palabra1,palabra2)]
            gold_standard+=[GS]
        Pearson_r=pearsonr(gold_standard,predicciones)[0]
        suma_Pearson_r+=Pearson_r*float(len(dataset))
        suma_numero_de_pares+=len(dataset)
        print nombre_dataset,"\t",len(dataset),"\t",round(Pearson_r,4)
    print "Prom.Pond.\t",suma_numero_de_pares,"\t",round(suma_Pearson_r/suma_numero_de_pares,4)









#print len(dataset1),dataset1

#dataset2=leer_dataset("./en/RG.txt")
#print len(dataset2),dataset2

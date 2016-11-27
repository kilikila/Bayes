#2.7.12
#coding:utf-8
import os
import re

from numpy import *
from nltk.corpus import stopwords

wordList=[]  #唯一字符列表

def filterWord(line):
    '''制定文本过滤规则,收集对象为：单词、常见符号$、！等、数字(非具体数值),需要过滤掉 数字、逗号等'''
    line=line.strip('\n') #去掉换行符

    wordlist=[]

    #过滤掉常见分词
    vocallist = re.split(r'\W|\d+', line)  #单词列表
    vocallist=[vocal.lower() for vocal in vocallist if vocal]
    wordlist.extend([vocal for vocal in vocallist if vocal not in stopwords.words('english')])

    #过滤掉  ， . 等语法中使用符号
    punctlist= re.split(r'\w|\s+', line)  #符号串列表

    '''
    punct_str=""
    for punct in punctlist:
        punct_str+=punct
    punctlist=list(punct_str)
    '''
    wordlist.extend([punct for punct in punctlist if punct not in [r',',r'.',r"'"] and punct])

    #过滤掉数字长度过大的数字字符串
    numberlist=re.split(r'\D+', line)  #数值列表
    wordlist.extend([numstr for numstr in numberlist if 0<len(numstr)<4 and numstr])

    return wordlist

def getWordList(url):
    '''将文本文件转化为字符列表'''
    file = open(url)
    wordlist=[]
    try:
        for line in file:
            if not line:continue
            wordlist.extend(filterWord(line))
    finally:
        file.close()

    return wordlist

def getDataVec(wordList,wordData):
    '''获取词汇向量w'''
    wordVec = [0]*len(wordList)
    for word in wordData:
        if word in wordList:
            wordVec[wordList.index(word)] += 1
    return array(wordVec)

def getTrainParam_pwc(trainDataSet):
    '''获取参数片p（w|c）'''
    trainDataSet=array(trainDataSet) #弄成numpy 对象
    dataNum= ones(len(trainDataSet[0]))
    dataSum = len(trainDataSet[0])
    for i in range(len(trainDataSet)):
        dataNum += trainDataSet[i]
        dataSum += sum(trainDataSet[i])
    p_wc = log(dataNum/dataSum)
    return p_wc


def getTrain(typeFileList):
    '''参数为各个分类类别的训练文件列表集合【[][][]】'''
    #计算各个类别的p(c)值
    numOfAll=sum([len(typelist) for typelist in typeFileList])+0.0
    pc_list = [len(typelist)/numOfAll for typelist in typeFileList]

    #---训练出各个类别的p(w|c)值---

    #获取词汇列表
    wordset=set([])
    for filelist in typeFileList:
        for fileurl in filelist:
            wordset=wordset|set(getWordList(fileurl))
    global wordList
    wordList=list(wordset)  #获取词汇列表

    #获取向量
    typeVecList=[]
    for filelist in typeFileList:
        typeVecs=[]
        for fileurl in filelist:
            typeVecs.append(getDataVec(wordList,getWordList(fileurl)))
        typeVecList.append(typeVecs)

    #获取各个类型的片p(w|c)值
    pwc_list=[]
    for typeVecs in typeVecList:
        pwc_list.append(getTrainParam_pwc(typeVecs))

    return pwc_list,pc_list

def getType(testVec, pwc_list,pc_list):
    '''判断类别待判断向量、p(w|c)值列表、p(c)值列表'''
    p=[]

    for (pwc,pc) in zip(pwc_list,pc_list):
        p.append(sum(testVec * pwc) + log(pc))
    return p.index(max(p))

def beginTest():
    '''确定训练样本和确定测试样本'''

    #进行训练
    trainDataSet=[]

    typelist1=[]
    filelist1 = os.listdir(trainDir_type1)
    for line in filelist1:
        filepath = os.path.join(trainDir_type1, line)
        if not os.path.isdir(filepath):
            typelist1.append(filepath)

    trainDataSet.append(typelist1)

    typelist2=[]
    filelist2 = os.listdir(trainDir_type2)
    for line in filelist2:
        filepath = os.path.join(trainDir_type2, line)
        if not os.path.isdir(filepath):
            typelist2.append(filepath)

    trainDataSet.append(typelist2)

    pwc_list, pc_list=getTrain(trainDataSet)


    # 开始测试

    rightNum=0.0
    wrongNum=0.0

    testlist1 = os.listdir(testDir_type1)
    for line in testlist1:
        filepath = os.path.join(testDir_type1, line)
        if not os.path.isdir(filepath):
            vec = getDataVec(wordList,getWordList(filepath))
            if getType(vec,pwc_list,pc_list)==0:
                rightNum+=1
            else:
                wrongNum+=1

    testlist2 = os.listdir(testDir_type2)
    for line in testlist2:
        filepath = os.path.join(testDir_type2, line)
        if not os.path.isdir(filepath):
            vec = getDataVec(wordList,getWordList(filepath))
            if getType(vec,pwc_list,pc_list)==1:
                rightNum+=1
            else:
                wrongNum+=1
    print "测试总数%.0f  成功数%.0f  失败数%.0f" % (rightNum+wrongNum,rightNum,wrongNum)
    print "正确率为%.3f %%" % ((rightNum*100)/(rightNum+wrongNum))


if __name__=="__main__":
    trainDir_type1 = r"E:\SrcCode\Naive_Bayes\testData\trainHam"
    trainDir_type2 = r"E:\SrcCode\Naive_Bayes\testData\trainSpam"

    testDir_type1 = r"E:\SrcCode\Naive_Bayes\testData\testHam"
    testDir_type2 = r"E:\SrcCode\Naive_Bayes\testData\testSpam"

    beginTest()
    pass
'''得到词汇表->进行学习->贝叶斯算法->测试'''


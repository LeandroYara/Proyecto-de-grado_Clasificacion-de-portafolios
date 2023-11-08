from csv import DictReader
from thefuzz import fuzz

def readTokenizedFiles():
    
    frameworkList = []
    scopusList = []
    
    with open("./data/tokenizedFrameworkData.csv", 'r', encoding = "utf8") as ff:
        dictReaderFramework = DictReader(ff)
        frameworkList = list(dictReaderFramework)
           
    with open("./data/tokenizedScopusData.csv", 'r', encoding = "utf8") as sf:
        dictReaderScopus = DictReader(sf)
        scopusList = list(dictReaderScopus)
                
    return (frameworkList, scopusList)

def calculateAffinity(frameworkList, scopusList):
    
    criteriaDict = {}
    documentDict = {}
    
    for document in frameworkList:
        criterion = document['Criterios']
        words = document['Words']
        if criterion not in criteriaDict:
            criteriaDict[criterion] = words
        else:
            for word in words.split():
                criterionWords = criteriaDict[criterion]
                if word not in criterionWords and word.isalpha():
                   criteriaDict[criterion] = " ".join([criterionWords, word])
                   
    for document in scopusList:
        title = document['Title']
        words = document['Words']
        documentDict[title] = {}
        for criterion in criteriaDict:
            documentDict[title][criterion] = fuzz.token_set_ratio(words, criteriaDict[criterion])
    
    return documentDict

fileTuple = readTokenizedFiles()
documentDict = calculateAffinity(fileTuple[0], fileTuple[1])
print(documentDict['40th International Conference on Performance and Capacity 2014 by CMG'])
        
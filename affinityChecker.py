from csv import DictReader
import csv

def readTokenizedFiles():
    
    frameworkList = []
    scopusList = []
    
    with open("data/tokenizedFrameworkData.csv", 'r', encoding = "utf8") as ff:
        dictReaderFramework = DictReader(ff)
        frameworkList = list(dictReaderFramework)
           
    with open("data/tokenizedScopusData.csv", 'r', encoding = "utf8") as sf:
        dictReaderScopus = DictReader(sf)
        scopusList = list(dictReaderScopus)
                
    return (frameworkList, scopusList)

def calculateAffinity(frameworkList, scopusList, titles):
    
    criteriaDict = {}
    documentDict = {}
    
    for document in frameworkList:
        criterion = document['Criterios']
        words = document['Words']
        if criterion not in criteriaDict:
            criteriaDict[criterion] = words
            titles.append(criterion)
        else:
            for word in words.split():
                criterionWords = criteriaDict[criterion]
                if word not in criterionWords and word.isalpha():
                   criteriaDict[criterion] = " ".join([criterionWords, word])
                   
    for document in scopusList:
        title = document['Title']
        wordsList = document['Words'].split(" ")
        documentDict[title] = {}
        for criterion in criteriaDict:
            criteriaList = criteriaDict[criterion].split(" ")
            common_elements = [element for element in criteriaList if element in wordsList]
            num_criteria_elements = len(criteriaList)
            documentDict[title][criterion] = round((len(common_elements) / num_criteria_elements) * 100, 2)
    
    return documentDict

def generateAffinityFile(documentDict):
    with open('data/documentAffinity.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
    
        writer.writerow(titles)
        
        for document in documentDict:
            dataList = []
            dataList.append(document)
            for criterion in documentDict[document]:
                dataList.append(documentDict[document][criterion])
            writer.writerow(dataList)
        
        file.close()

fileTuple = readTokenizedFiles()
titles = ["Title"]
documentDict = calculateAffinity(fileTuple[0], fileTuple[1], titles)
generateAffinityFile(documentDict)
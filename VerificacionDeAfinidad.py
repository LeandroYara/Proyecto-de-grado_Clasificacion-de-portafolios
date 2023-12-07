from csv import DictReader
import csv

def readTokenizedFiles():
    
    print("Leyendo archivos...")
    frameworkList = []
    inputList = []
    
    with open("data/transformedFrameworkData.csv", 'r', encoding = "utf8") as ff:
        dictReaderFramework = DictReader(ff)
        frameworkList = list(dictReaderFramework)
           
    with open("data/transformedInputData.csv", 'r', encoding = "utf8") as inf:
        dictReaderInput = DictReader(inf)
        inputList = list(dictReaderInput)
                
    return (frameworkList, inputList)

def calculateAffinity(frameworkList, inputList, titles):
    
    print("Calculando afinidad...")
    
    categoryDict = {}
    documentDict = {}
    
    for document in frameworkList:
        category = document['Category']
        words = document['Words'].split()
        if category not in categoryDict:
            categoryDict[category] = words
            titles.append(category)
        else:
            for word in words:
                categoryWords = categoryDict[category]
                if word not in categoryWords and word.isalpha():
                   categoryDict[category].append(word)
                   
    for document in inputList:
        title = document['Title']
        wordsList = document['Words'].split(" ")
        documentDict[title] = {}
        for category in categoryDict:
            categoriesList = categoryDict[category]
            common_elements = [element for element in categoriesList if element in wordsList]
            num_categories_elements = len(categoriesList)
            documentDict[title][category] = round((len(common_elements) / num_categories_elements) * 100, 2)
    
    return documentDict

def generateAffinityFile(documentDict):
    
    print("Escribiendo archivo de afinidad...")
    
    with open('data/inputAffinity.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
    
        writer.writerow(titles)
        
        for document in documentDict:
            dataList = []
            dataList.append(document)
            for category in documentDict[document]:
                dataList.append(documentDict[document][category])
            writer.writerow(dataList)
        
        file.close()

fileTuple = readTokenizedFiles()
titles = ["Title"]
documentDict = calculateAffinity(fileTuple[0], fileTuple[1], titles)
generateAffinityFile(documentDict)
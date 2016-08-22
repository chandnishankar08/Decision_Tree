"""
    Implementation of decision tree using ID3 algorithm
    Author:Chandni Shankar
"""
import math, sys
from collections import Counter, deque
from random import randint
from copy import copy, deepcopy

def removekey(originalDict, key):
    """
        Utility function to remove the key from a copy of the dictionary to preserve original Dictionary
    """
    duplicateDict = dict(originalDict)
    del duplicateDict[key]
    return duplicateDict

                       
def postPrune(tree, l, k, valData):
    """
        Function to prune the decision tree that was built in order to find a better tree if possible (Smaller trees are better)
    """
    BestTree = tree
    BestAccuracy = getAccuracy(BestTree, valData)
    for i in range(1,int(l)):
        tempTree = deepcopy(BestTree)
        m = randint(1,int(k))
        for j in range(1,m):
            n = len(tempTree.nodes)
            if n > 1:
                p = randint(0,(n-1))
            else:
                p = 0
            if p < n:
                node = tempTree.nodes[p]
                attrListData = Counter()
                for nodeEdge in node.adj:
                    for instance in nodeEdge.instanceList:
                        attrListData[instance.label] += 1
                maxLabel = max(attrListData, key=attrListData.get)
                parent = node.parent
                if parent == None:
                    tempTree.remove()
                    tempTree.addLeaf(None, maxLabel)
                else:
                    for edge in parent.adj:
                        if edge.getToNode(parent) == node :
                            tempTree.removeNode(node, edge)
                            tempTree.addLeaf(edge, maxLabel)
        tempAccuracy = getAccuracy(tempTree, valData)
        if tempAccuracy >= BestAccuracy:
            BestTree = tempTree
            BestAccuracy = tempAccuracy
    return BestTree
            
def predictLabel(tree, testData):
    """
        Function to predict the label of the testInstance using the decision tree built
    """
    if tree.treeLeaf:
        return tree.root.name
    root = tree.getRoot()
    while root:
        testVal = testData.features[root.num]
        for edge in root.adj:
            if edge.name == testVal:
                edgeToUse = edge
        nextNode = edgeToUse.getToNode(root)
        if nextNode == None:
            return edgeToUse.getLeaf()
        root = nextNode

def getAccuracy(tree, validationData):
    """
        Function to calculate accuracy of the decision tree. The percentage of no.of correctly predicted values are returned
    """
    accuracy = 0.0
    for dataNum in range(1,len(validationData)):
        data = Instances(validationData[dataNum],dataNum)
        actualLabel = data.label
        predictedLabel = predictLabel(tree, data)
        if actualLabel == predictedLabel:
            accuracy += 1.0
    return (accuracy/len(validationData)*100)
    
def entropy(trainData, entropyI):
    """
        Function to calculate entropy
        if Variance impurty heuristic (entropyI!='e') entropy = k0*k1/k*k, k = total samples, k0 = class 0 samples, k1 = class 1 samples
        if Information gain heuristic (entropyI='e'), entropy = Sum(-(k0/k)log2(k0/k)-(k1/k)log2(k1/k))
    """
    c = Counter()
    for instance in trainData:
        c[instance.label] += 1
    total = sum(c.values())
    if entropyI == 'e':
        entr = 0.0
    else:
        entr = 1.0
    if total == 0:
        return 0.0
    else:
        for labelVal, count in c.iteritems():
            if entropyI == 'e':
                entr += (-float(count)/float(total) * (math.log(count,2) - math.log(total,2)))
            else:
                entr *= (float(count)/float(total))
    return entr

def varImputity(trainData):
    c = Counter()
    for instance in trainData:
        c[instance.label] += 1
    total = sum(c.values())
    if total == 0:
        return 0.0
    else:
        entr = ((float(c[1])/float(total))* (float(c[0])/float(total)))
    return entr

def informationGain(data, attrNum, parentEntr, entropyI):
    """
        Function to calculate information gain        
    """
    infoGain = parentEntr
    attrDict = dict()
    for instanceObj in data:
        if instanceObj.features[attrNum] in attrDict: #Creating a dictionary mapping each Attribute values and list of instances for this the attr val is true
            attrDict[instanceObj.features[attrNum]].append(instanceObj)
        else:
            attrDict[instanceObj.features[attrNum]] = [instanceObj]  
    for attrVal, attrDataList in attrDict.iteritems():
        if entropyI == 'e':
            infoGain += -float(len(attrDataList))/float(len(data))* entropy(attrDataList, entropyI)
        else:
            infoGain -= float(len(attrDataList))/float(len(data))* varImputity(attrDataList)
    return infoGain, attrDict 

class Instances:
    """
        Class to create instances with features in array and actual label of each instance
    """
    def __init__(self, data, idNum):
        dataLen = len(data)
        self.label = data[-1]
        self.features = data[:-1]
        self.id = idNum
        
class Node:
    """
        Defines a node of a tree, the Attribute Name is added a node
    """
    def __init__(self, name, num):
        self.name = name
        self.parent = None
        self.parentLabel = None
        self.seen = True
        self.adj = []
        self.num = num
        
class Edge:
    """
        Defines a edge from a node of a tree, the Attribute's values are added as an edge
    """
    def __init__(self, fromNode, name, objList, features):
        self.name = name
        self.fromNode = fromNode
        self.toNode = None
        self.instanceList = objList
        self.featureList = features
        self.leaf = False
        self.leafVal = None

    def addToNode(self, toNode):
        """
            adds the other node to the edge
        """
        self.toNode = toNode
  
    def addLeaf(self, leaf):
        """
            adds a leaf to the edge as the other end
        """
        self.leafVal = leaf
        self.leaf = True

    def getToNode(self, node):
        if self.fromNode == node:
            return self.toNode
        else:
            return None
        
    def removeNode(self, node):
        self.toNode = None

    def getLeaf(self):
        return self.leafVal
    
    def isLeaf(self):
        return self.leaf
        

class Tree(object):
    """
        Defines a tree as a graph with attributes as vertices and attribute values as edges
    """
    def __init__(self):
        self.nodes = []
        self.leaves = []
        self.size = 0
        self.treeLeaf = False
        self.root = None
        
    def addNode(self, node):
        if self.size == 0:
            self.root = node
        self.nodes.append(node)        
        self.size += 1

    def removeNode(self, node, edge):
        self.nodes.remove(node)
        self.size -= 1
        edge.removeNode(node)

    def remove(self):
        self.nodes = []
        self.leaves = []
        self.size = 0
        self.treeLeaf = False
        self.root = None
        
    def getRoot(self):
        return self.root
    
    def addEdge(self, initNode, edge):
        initNode.adj.append(edge)

    def addLeaf(self, edge, name):
        if self.size == 0:
            self.root = Node(name, -1)
            self.treeLeaf = True
            self.leaves.append(name)
        else:
            edge.addLeaf(name)
            self.leaves.append(name)
        self.size += 1

    def __deepcopy__(self, memo):
        clsInit = self.__class__
        newObject = clsInit.__new__(clsInit)
        memo[id(self)] = newObject
        for k, v in self.__dict__.items():
            setattr(newObject, k, deepcopy(v, memo))
        return newObject
   
    def buildTree(self, data, features, entropyI):
        """
            Function to build the tree
        """
        toExplore = []
        toExplore.append("Start") # Assigning Start to check if tree is null but preserving reusability
        while toExplore: # Building tree till all added edges are visited once
            curEdge = toExplore.pop()
            if curEdge == "Start":
                trainData = data
                unExploredFeat = features
                predecessorLabel = None
            else:
                trainData = curEdge.instanceList  # While atleast one node is present in the tree
                unExploredFeat = curEdge.featureList
                predecessorLabel = curEdge.fromNode.parentLabel
            if not trainData: # Checking if all values of the given data are the same to add a leaf node. A leaf could also be the whole tree.
                self.addLeaf(curEdge, predecessorLabel)
            else:
                checkLeaf = Counter()
                for instance in trainData:
                    checkLeaf[instance.label] += 1
                mostFreqLabel = max(checkLeaf, key=checkLeaf.get)
                if not unExploredFeat or len(checkLeaf) == 1: #If leaf node or attribute list is empty, assigning label with max value, if Leaf node, the only label will have max value
                    self.addLeaf(curEdge, mostFreqLabel)
                else:
                    if entropyI == 'e':
                        c = entropy(trainData, entropyI) #Entropy of current node to calculate information gain
                    else:
                        c = varImputity(trainData)
                    attrValDict = {}
                    allInfoGain = {}                    
                    for attrNum, name in unExploredFeat.iteritems(): #For each unexplored attr along the path, data for that attribute for each of its values are placed in a dict
                        attrInfoGain,attrDict  = informationGain(trainData, attrNum, c, entropyI)
                        attrValDict[attrNum] = attrDict
                        allInfoGain[attrNum] = attrInfoGain
                    attr = max(allInfoGain, key = allInfoGain.get) #attribute with max infomation gain     
                    bestAttrNode = Node(unExploredFeat[attr], attr)
                    self.addNode(bestAttrNode) #Adding best attribute to the tree
                    bestAttrNode.parentLabel = mostFreqLabel
                    if curEdge != "Start":
                        curEdge.addToNode(bestAttrNode) #Assigning node to the edge adding and assigning parent for this node
                        bestAttrNode.parent = curEdge.fromNode
                    attrDict = attrValDict[attr]
                    attrFeatList = removekey(unExploredFeat,attr) 
                    for key, objList in attrDict.iteritems():
                        edge = Edge(bestAttrNode, key, objList, attrFeatList)  #Add edge to the tree for each value of the attribute
                        self.addEdge(bestAttrNode, edge)
                        toExplore.append(edge)
                        
    def printTree(self):
        """
            Prints the tree in the Heirarchial form, horizontally
        """
        if self.size == 1:
            print self.root.name
        else:
            root = self.getRoot()
            stack = []
            levelIndictor = 0
            for edge in root.adj:
                stack.append((edge,levelIndictor))
            while stack:
                edge, levelIndicator = stack.pop()
                print (levelIndicator*'| '),
                if edge.isLeaf():
                    print '%s = %s : %s' %(edge.fromNode.name ,edge.name,edge.leafVal)
                else:
                    print '%s = %s : ' % (edge.fromNode.name, edge.name)
                    levelIndicator += 1
                    nextNode = edge.getToNode(edge.fromNode)
                    for nodeEdge in nextNode.adj:
                        stack.append((nodeEdge,levelIndicator))
        
def main(argv):
    l = argv[0]
    k = argv[1]
    trainFile = argv[2]
    validationFile = argv[3]
    testFile = argv[4]
    toPrint = argv[5]
    lineNo = 0
    trainData = []
    feature = {}
    for line in open(trainFile):
        curLine = line.rstrip('\n').split(',')
        if lineNo == 0:
            for i in range(len(curLine)-1):
                feature[i] = curLine[i]
        else:
            trainData.append(Instances(curLine, lineNo))
        lineNo += 1
    decisionTreeIG = Tree()
    decisionTreeVI = Tree()
    decisionTreeIG.buildTree(trainData, feature, 'e')
    validnData = [line.rstrip('\n').split(',') for line in open(validationFile)]
    testData = [line.rstrip('\n').split(',') for line in open(testFile)]
    accuracyIG = getAccuracy(decisionTreeIG, testData)
    print "Before pruning:"
    print "Accuracy using Information gain heuristic:", accuracyIG
    if toPrint == 'yes':
        decisionTreeIG.printTree()

    decisionTreeVI.buildTree(trainData, feature, 'v')
    accuracyVI = getAccuracy(decisionTreeVI, testData)
    print "Accuracy using Variance Impurity heuristic:", accuracyVI
    if toPrint == 'yes':
        decisionTreeVI.printTree()
    print "After pruning:"
    prunedTreeIG = postPrune(decisionTreeIG, l, k, validnData)
    prunedAccuracyIG = getAccuracy(prunedTreeIG, testData)
    if toPrint == 'yes':
        prunedTreeIG.printTree()
    print "Accuracy using Information gain heuristic:", prunedAccuracyIG
    prunedTreeVI = postPrune(decisionTreeVI, l, k, validnData)
    prunedAccuracyVI = getAccuracy(prunedTreeVI, testData)
    print "Accuracy using Variance Impurity heuristic:", prunedAccuracyVI
    if toPrint == 'yes':
        prunedTreeVI.printTree()    
            
if __name__ == "__main__":
   main(sys.argv[1:])
    

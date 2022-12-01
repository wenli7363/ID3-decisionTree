# 读数据集合，构造数据集
import operator
import math
import treeplot

def read_dataset(filename):
    fr = open(filename, 'r')
    all_lines = fr.readlines()  # list形式,每行为1个str
    dataset = []
    for line in all_lines[1:]:
        line = line.strip().split('\t')  # 以\t为分割符拆分列表
        dataset.append(line)
    dataset.pop()  # 删掉最后文件尾部的一些杂项
    # 将字符串元素转换为float型
    result = []
    for item in dataset:
        tmp = list(map(float,item[:-1]))
        tmp.append(int(item[-1]))
        result.append(tmp)
    return result


# -----------------------------------------------------------------------
def classify(inputTree, featLabels, testVec):
    """
    输入：决策树，分类标签，测试数据
    输出：决策结果
    描述：跑决策树
    """
    firstStr = list(inputTree.keys())[0]  # 取出决策树根节点
    value = float(firstStr[4:])
    secondDict = inputTree[firstStr]       # 取出下一层所有分支
    featIndex = featLabels.index(firstStr[:3]) # 求根节点属性，在属性集的下标
    classLabel = 0
    for key in secondDict.keys():  # 遍历下一层分支{属性1：{下一层子节点}，属性2：{下一层子节点}}
        if key == '小于':
            if testVec[featIndex] <= value: # 如果该测试集的属性与分支的第二层属性匹配上
                if type(secondDict[key]).__name__ == 'dict': # 如果匹配上的第二层属性的下一层还是字典（说明还能再分支）
                    classLabel = classify(secondDict[key], featLabels, testVec)  # 递归
                else:   # 不能继续分支，是叶子节点
                    classLabel = secondDict[key]
        if key == '大于':
            if testVec[featIndex] > value: # 如果该测试集的属性与分支的第二层属性匹配上
                if type(secondDict[key]).__name__ == 'dict': # 如果匹配上的第二层属性的下一层还是字典（说明还能再分支）
                    classLabel = classify(secondDict[key], featLabels, testVec)  # 递归
                else:   # 不能继续分支，是叶子节点
                    classLabel = secondDict[key]
    return classLabel

def classifytest(inputTree, featLabels, testDataSet):
    """
    输入：决策树，分类标签，测试数据集
    输出：决策结果
    描述：跑决策树
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll

def majorCnt(labels):
    labelCount = dict()   # 创建一个字典，记录每个类别的样本数
    for value in labels:
        if value not in labelCount.keys():
            labelCount[value] = 0
        labelCount[value] += 1
    maxCount = sorted(labelCount.items(),key=operator.itemgetter(1),reverse=True)
    return maxCount[0][0]  # 取样本数最多的类别

# 获取数据集中，在axis维度上值为value的样本子集
def splitDataSet(dataset,axis,value):
    smallDataSet = []
    greatDataSet = []
    # 遍历每一条数据，找到符合要求的数据，同时去掉在axis属性上的取值
    for featVec in dataset:
        if featVec[axis] <= value:
            smallDataSet.append(featVec)
        else:
            greatDataSet.append(featVec)
    return smallDataSet,greatDataSet

# 计算信息熵
def calEnt(dataset):
    numEntries = len(dataset)  # 样本集总数
    labelCount = dict()  # 记录每个标签有多少样本
    for featVec in dataset:
        if featVec[-1] not in labelCount.keys():
            labelCount[featVec[-1]] = 0
        labelCount[featVec[-1]] += 1
    Ent = 0
    for key in labelCount.keys():
        p = float(labelCount[key])/numEntries  # pi
        Ent -= p * math.log(p,2)  # -sum ( pi * log2(pi))
    return Ent

# 对于连续值属性，同一个属性中，选一个最优二分点
def findBest_mid(dataset,axis,T):
    baseEnt = calEnt(dataset)
    bestIG = 0
    best_mid = 0
    for t in T:
        smallDataSet, greatDataSet = splitDataSet(dataset, axis, t)
        p1 = len(smallDataSet) / float(len(dataset))
        p2 = len(greatDataSet) / float(len(dataset))
        newEnt = p1 * calEnt(smallDataSet) + p2 * calEnt(greatDataSet)
        newIG = baseEnt - newEnt
        if newIG > bestIG:
            bestIG = newIG
            best_mid = t
    return best_mid


# 获取最好的属性划分  ID3算法
def chooseBestFeature(dataset):
    numAttribute = len(dataset[0])-1 # 当前dataset中的属性个数，减1是因为最后一个是标签
    baseEntropy = calEnt(dataset)  # 计算当前dataset的信息熵
    bestInfoGain = 0               # 初始化信息增益为0
    bestFeature = -1               # 初始化最优属性为-1
    best_mid = -1

    for i in range(numAttribute):
        featList = [example[i] for example in dataset]   # 取出属性i的所有取值
        uniqueVals = set(featList)   # 去重，得到分支的个数
        T = []  # 连续值离散化

        # 连续值求中点
        sortedV = sorted(uniqueVals)
        for j in range(len(sortedV)-1):
            T.append((sortedV[j]+sortedV[j+1])/2.0)

        # for value in T:
        #     subDataSet = splitDataSet(dataset,i,value)
        #     p = len(subDataSet)/float(len(dataset))  # |Dj|/|D|
        #     newEntropy += p * calEnt(subDataSet)     # sum( |Dj|/|D| * info(Dj))

        # 计算样本子集的信息熵
        mid = findBest_mid(dataset,i,T)  # 得到属性i中最优的划分值
        smallDataSet, greatDataSet = splitDataSet(dataset,i,mid)
        p1 = len(smallDataSet)/float(len(dataset))
        p2 = len(greatDataSet)/float(len(dataset))
        newEntropy = p1 * calEnt(smallDataSet) + p2 * calEnt(greatDataSet)
        # 计算信息增益
        infoGain = baseEntropy - newEntropy
        # 找最大信息增益
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
            best_mid = mid

    return bestFeature,best_mid


def TreeGenerate(dataset:list, Attributes):
    # 取数据集最后一列构成类别列表
    labels = [x[-1] for x in dataset]

    # 第一种情况，当样本集全部元素都是相同的类别，直接分为这一类
    if labels.count(labels[0]) == len(dataset):
        return labels[0]

    # # 第二种情况，属性集为空 ，返回D中样本数最多的类别
    # # 在连续值的条件，不存在，因为不删属性集
    # if len(dataset[0]) == 1 :
    #     return majorCnt(labels)

    bestFeature,BestMid = chooseBestFeature(dataset) # 从样本集中选最好的属性,以及当前属性在某点的最优二分
    bestFeatureAttr = Attributes[bestFeature] +"<{}".format(BestMid)
    myTree = {bestFeatureAttr:{}}

    # del(Attributes[bestFeature])  # 删掉属性集中用过的属性
    # 下面这个适合离散属性，本次为连续值，只作二分
    # featValue = [x[bestFeature] for x in dataset] # 最好属性的属性值
    # uniqueVal = set(featValue) # 去重，就能得到节点在这一属性上所有的分支的属性值（离散的条件下）

    #递归的构造子树
    # for value in uniqueVal:
    #     subAttributes = Attributes[:] # 子属性集合
    #     subDataset = splitDataSet(dataset,bestFeature,value)  # 输入数据集，获得数据集中，在bestFeature上取值为value的样本子集
    #     myTree[bestFeatureAttr][value] = TreeGenerate(subDataset,subAttributes)

    # left sub tree  value < x; right subTree is value >x
    smallDataSet,greatDataSet = splitDataSet(dataset,bestFeature,BestMid)
    if len(smallDataSet) != 0:
        myTree[bestFeatureAttr]['小于'] = TreeGenerate(smallDataSet,Attributes)
    if len(greatDataSet) != 0:
        myTree[bestFeatureAttr]['大于'] = TreeGenerate(greatDataSet,Attributes)
    return myTree

def calCorrectRate(testset,classLabelAll):
    realLabel = []
    correct = 0
    for item in testset:
        realLabel.append(item[-1])
    for i in range(len(realLabel)):
        if realLabel[i] == classLabelAll[i]:
            correct += 1

    return correct/float(len(realLabel))


if __name__ == '__main__':
    filename="./traindata.txt"
    dataset=read_dataset(filename)

    Attributes = ['属性1','属性2','属性3','属性4']
    myTree = TreeGenerate(dataset,Attributes)
    print(myTree)
    # treeplot.ID3_Tree(myTree)
    testset = read_dataset("./testdata.txt")
    classLabelAll = classifytest(myTree,Attributes,testset) # 测试集通过决策树的分类标签
    result = calCorrectRate(testset,classLabelAll)
    print(result)

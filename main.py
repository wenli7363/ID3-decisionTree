# 读数据集合，构造数据集
import operator
import math
import treeplot

def read_dataset(filename):
    """
    输入：文件路径
    输出：数据集列表
    描述：数据集处理，读入txt文件，输出二维的列表，每一个元素是一条包含分类结果的item
    """
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


def classify(inputTree, featLabels, testVec):
    """
    输入：决策树，属性标签->[’属性1‘，’属性2‘，’属性3‘，‘属性4’]，数据集中的一条测试数据（一维list）
    输出：这一条数据通过决策树后的决策结果：1，2，3中的某种情况
    描述：跑决策树
    """
    firstStr = list(inputTree.keys())[0]  # 取出决策树根节点
    value = float(firstStr[4:])           # 这里的value指每一个节点后面接的最优划分值，比如根节点(firstStr)为”属性3<2.45“，那么value就是第4个字符以后的数字

    secondDict = inputTree[firstStr]       # 取出下一层所有分支，即{firstStr:{}}中，键firstStr对应的值
    featIndex = featLabels.index(firstStr[:3]) # 求根节点属性，在属性标签集中的下标，比如”属性3<2.45“，用字符串切片取出”属性3“，然后获得在属性集合下标为2
    classLabel = 0

    for key in secondDict.keys():  # 遍历下一层分支{'小于'：{下一层子节点}，'大于'：{下一层子节点}}
        if key == '小于':
            if testVec[featIndex] <= value: # 如果该测试集数据 <= 最优化分值value
                if type(secondDict[key]).__name__ == 'dict': # 如果下一层还是字典（说明还能再分支）
                    classLabel = classify(secondDict[key], featLabels, testVec)  # 递归
                else:   # 不能继续分支，是叶子节点
                    classLabel = secondDict[key]
        if key == '大于':
            if testVec[featIndex] > value: # 如果该数据 >= value
                if type(secondDict[key]).__name__ == 'dict': # 如果下一层还是字典（说明还能再分支）
                    classLabel = classify(secondDict[key], featLabels, testVec)  # 递归
                else:   # 不能继续分支，是叶子节点
                    classLabel = secondDict[key]
    return classLabel

def classifytest(inputTree, featLabels, testDataSet):
    """
    输入：决策树，属性名集合->[’属性1‘，’属性2‘，’属性3‘，‘属性4’]，测试数据集（二维list）
    输出：整个数据集的决策结果 ->list
    描述：整个数据集跑决策树
    """
    classLabelAll = []  # 存整个数据集的分类结果
    for testVec in testDataSet:     # 取出数据集中的每一条记录跑决策树
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll

def majorCnt(labels):
    """
    输入：当前分类结果的集合
    输出：数量最多的一种分类结果
    描述：返回当前数据集中数量最多的一种分类
    """
    labelCount = dict()   # 创建一个字典，记录每个类别的样本数
    for value in labels:
        if value not in labelCount.keys():
            labelCount[value] = 0
        labelCount[value] += 1
    maxCount = sorted(labelCount.items(),key=operator.itemgetter(1),reverse=True)   # 排序
    return maxCount[0][0]  # 取样本数最多的类别

def splitDataSet(dataset,axis,value):
    """
    输入：数据集，维度，值
    输出：分出两个子集
    描述：获取数据集中，在axis维度上值大于、小于value的样本子集
    """
    smallDataSet = []
    greatDataSet = []
    # 遍历每一条数据，找到符合要求的数据，同时去掉在axis属性上的取值
    for featVec in dataset:
        if featVec[axis] <= value:
            smallDataSet.append(featVec)
        else:
            greatDataSet.append(featVec)
    return smallDataSet,greatDataSet

def calEnt(dataset):
    """
    输入：数据集
    输出：float，信息熵大小
    描述：计算当前集合的信息熵大小
    """
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
    """
    输入：数据集，维度，离散化后的可能的分类值点
    输出：最优的分类值
    描述：对于同一连续属性上，找到一个最优的分类值，也就是求某属性<??最好
    """
    baseEnt = calEnt(dataset)  # 数据集二分前的信息熵
    bestIG = 0                 # 最优信息增益值
    best_mid = 0                # 最优分类值

    # 遍历可能分类值集合T，计算同一属性在不同的分类点下的信息增益
    for t in T:
        smallDataSet, greatDataSet = splitDataSet(dataset, axis, t)
        p1 = len(smallDataSet) / float(len(dataset))
        p2 = len(greatDataSet) / float(len(dataset))
        newEnt = p1 * calEnt(smallDataSet) + p2 * calEnt(greatDataSet)
        newIG = baseEnt - newEnt
        if newIG > bestIG:
            bestIG = newIG      # 更新最优信息增益
            best_mid = t
    return best_mid


# 获取最好的属性划分  ID3算法
def chooseBestFeature(dataset):
    """
    输入：输入数据集
    输出：当前数据集下的最好的分类属性
    """
    numAttribute = len(dataset[0])-1 # 当前dataset中的属性个数，减1是因为最后一个是标签
    baseEntropy = calEnt(dataset)  # 计算当前dataset的信息熵
    bestInfoGain = 0               # 初始化信息增益为0
    bestFeature = -1               # 初始化最优属性为-1
    best_mid = -1

    # 遍历所有属性，找出最大信息增益对应的属性
    for i in range(numAttribute):
        featList = [example[i] for example in dataset]   # 取出属性i的所有取值
        uniqueVals = set(featList)   # 去重，得到分支的个数
        T = []  # 连续值离散化列表

        # 连续值求中点
        sortedV = sorted(uniqueVals)
        for j in range(len(sortedV)-1):
            T.append((sortedV[j]+sortedV[j+1])/2.0)

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


def TreeGenerate(dataset, Attributes):
    """
    输入：训练集，属性名集合
    输出：一个决策树{根节点：{左子树：{}，右子树；{}}}
    描述：生成一个决策树
    """

    # 取数据集最后一列构成分类后的结果
    labels = [x[-1] for x in dataset]

    # 限定深度
    global branch,maxBranch         # branch为产生分支的节点数（决策次数），maxBranch表示允许最多做几次决策

    if branch >= maxBranch :        # 如果到达限定深度，直接返回当前集合中频度最高的种类
        return majorCnt(labels)

    if labels.count(labels[0]) == len(dataset): # 如果数据集都是同种类别，就直接定为这个种类
        return labels[0]

    bestFeature,BestMid = chooseBestFeature(dataset) # 从样本集中选最好的属性,以及当前属性在某点的最优二分值
    bestFeatureAttr = Attributes[bestFeature] +"<{}".format(BestMid)    # 字符串拼接
    myTree = {bestFeatureAttr:{}}       # 以当前属性为分支节点

    # 开始递归+二分，'大于'为左子树，'小于'为右子树
    smallDataSet,greatDataSet = splitDataSet(dataset,bestFeature,BestMid)
    if len(greatDataSet) != 0:
        branch += 1             # 每递归生成一个左子树就说明做了一次决策，决策次数+1
        myTree[bestFeatureAttr]['大于'] = TreeGenerate(greatDataSet,Attributes)
    if len(smallDataSet) != 0:
        myTree[bestFeatureAttr]['小于'] = TreeGenerate(smallDataSet,Attributes)

    return myTree

def calCorrectRate(testset,classLabelAll):
    """
    输入：测试集，测试集通过决策树后的分类结果集合
    输出：float 分类的正确率
    描述：比较测试集的真实分类结果和决策树结果，计算出分类的正确率
    """
    realLabel = []      # 数据集真正的分类结果
    correct = 0
    for item in testset:
        realLabel.append(item[-1])

    for i in range(len(realLabel)):     # 遍历比较
        if realLabel[i] == classLabelAll[i]:
            correct += 1

    return correct/float(len(realLabel))


if __name__ == '__main__':
    dataset=read_dataset("./traindata.txt")     # 加载样本集
    Attributes = ['属性1','属性2','属性3','属性4'] # 添加属性标签

    # 限定深度操作
    branch,maxBranch =0,4       # branch当前已经做的决策次数，maxBranch为允许最多做几次决策（就是分支点数）
# ------------------------  生成决策树  -----------------------
    myTree = TreeGenerate(dataset,Attributes)
    print(myTree)

# --------------------------  结果分析  ---------------------------

    # 训练集在决策树上的分类正确率
    dataset_label = classifytest(myTree,Attributes,dataset) # 训练集通过决策树分类产生的标签
    dataset_result = calCorrectRate(dataset,dataset_label)
    print("训练集在决策树上的分类正确率：{}".format(dataset_result))

    # 测试集在决策树上的分类正确率
    testset = read_dataset("./testdata.txt")        # 加载测试集
    testset_label = classifytest(myTree,Attributes,testset) # 测试集通过决策树产生的分类标签
    testset_result = calCorrectRate(testset,testset_label)
    print("测试集在决策树上的分类正确率：{}".format(testset_result))

    # 决策树可视化
    treeplot.ID3_Tree(myTree)
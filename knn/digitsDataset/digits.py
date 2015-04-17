import numpy
import math
import heapq

def dist(a, b):
    d = a - b
    return math.sqrt(numpy.dot(d, d))

def classify(x, train_features, train_labels, k):
    dist_features = numpy.array(list(map(lambda feature: dist(x, feature), train_features)))
    return list(map(lambda elem: train_labels[elem], numpy.argsort(dist_features)[:k]))

def knn(k, train_features, train_labels, val_features):
    labels = []
    for val_feature in val_features:
        label_list = classify(val_feature, train_features, train_labels, k)
        vote, i = {}, 0
        for label in label_list:
            if label in vote:
                vote[label].append(i)
            else:
                vote[label] = [i]
            i += 1
        mode, len_mode, sum_mode = 0, 0, 0
        for label in vote.keys():
            l, s = len(vote[label]), sum(vote[label])
            if l > len_mode:
                mode, len_mode, sum_mode = label, l, s
            elif (l == len_mode) and (s < sum_mode):
                mode, sum_mode = label, s
        labels.append(mode)
    return labels

def choose(k, train_features, train_labels, val_features, val_labels):
    result = knn(k, train_features, train_labels, val_features)
    i, error = 0, 0
    for label in val_labels:
        if label != result[i]:
            error += 1
        i += 1
    print("k = " + str(k) + ", error rate = " + str(error / i))
    file_name = "digitsOutput" + str(k) + ".csv"
    with open(file_name, "w") as temp_file:
        temp_file.write("\n".join(map(lambda label: str(label), result[:1000])))

def main():
    with open("trainFeatures.csv") as temp_file:
        lines = temp_file.read().split("\n")
        train_features = [numpy.array(list(float(num) for num in line.split(',') if num)) for line in lines]
    with open("trainLabels.csv") as temp_file:
        lines = temp_file.read().split("\n")
        train_labels = [int(line) for line in lines]
    with open("valFeatures.csv") as temp_file:
        lines = temp_file.read().split("\n")
        val_features = [numpy.array(list(float(num) for num in line.split(',') if num)) for line in lines]
    with open("valLabels.csv") as temp_file:
        lines = temp_file.read().split("\n")
        val_labels = [int(line) for line in lines]
    choose(1, train_features, train_labels, val_features, val_labels)
    choose(2, train_features, train_labels, val_features, val_labels)
    choose(5, train_features, train_labels, val_features, val_labels)
    choose(10, train_features, train_labels, val_features, val_labels)
    choose(25, train_features, train_labels, val_features, val_labels)
    print("The best k is 1 or 2 (the same under my tie breaking rule)")
    print("Use k = 1 or 2 on test set")
    with open("testFeatures.csv") as temp_file:
        lines = temp_file.read().split("\n")
        test_features = [numpy.array(list(float(num) for num in line.split(',') if num)) for line in lines]
    result = knn(1, train_features, train_labels, test_features)
    with open("digitsOutput.csv", "w") as temp_file:
        temp_file.write("\n".join(map(lambda label: str(label), result[:1000])))

main()
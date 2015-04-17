import numpy
import random
import math

def entropy(p):
    if (p == 0) or (p == 1):
        return 0
    return -p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)

def compute_goodness(threshold, index, train_features, train_labels):
    left_labels, right_labels = [], []
    l = len(train_labels)
    for i in range(l):
        temp_feature, temp_label = train_features[i], train_labels[i]
        if temp_feature[index] <= threshold:
            left_labels.append(temp_label)
        else:
            right_labels.append(temp_label)
    left_len, left_sum = len(left_labels), sum(left_labels)
    right_len, right_sum = len(right_labels), sum(right_labels)
    left_val, right_val, val = float(left_sum) / left_len, float(right_sum) / right_len, \
        float(sum(train_labels)) / l
    return entropy(val) - (left_len / l * entropy(left_val) + right_len / l * entropy(right_val))

def build_decision_tree(train_features, train_labels):
    s, n = float(sum(train_labels)), len(train_labels)
    if entropy(s / n) < 0.1:
        return int(round(s / n))
    l = len(train_features[0])
    indices = list(range(l))
    if l == 0:
        return int(round(s / n))
    if l > 8:
        sampling_indices = sorted(random.sample(indices, 8))
    else:
        sampling_indices = indices
    max_goodness, max_threshold, max_feature = -float('inf'), 0, sampling_indices[0]
    for index in sampling_indices:
        index_features = sorted(set(map(lambda feature: feature[index], train_features)))
        thresholds = \
            [(index_features[i] + index_features[i + 1]) / 2 for i in range(len(index_features) - 1)]
        for threshold in thresholds:
            goodness = compute_goodness(threshold, index, train_features, train_labels)
            if goodness > max_goodness:
                max_goodness, max_threshold, max_feature = goodness, threshold, index
    left_features, left_labels, right_features, right_labels = [], [], [], []
    for i in range(n):
        temp_feature, temp_label = train_features[i], train_labels[i]
        if temp_feature[max_feature] <= max_threshold:
            left_features.append(temp_feature)
            left_labels.append(temp_label)
        else:
            right_features.append(temp_feature)
            right_labels.append(temp_label)
    if len(left_labels) == 0:
        return build_decision_tree(right_features, right_labels)
    if len(right_labels) == 0:
        return build_decision_tree(left_features, left_labels)
    left_tree = build_decision_tree(left_features, left_labels)
    right_tree = build_decision_tree(right_features, right_labels)
    return (max_feature, max_threshold), left_tree, right_tree
        

def bagging(train_features, train_labels):
    n = len(train_labels)
    random_indices = numpy.random.choice(n, size = n, replace = True)
    return list(map(lambda index: train_features[index], random_indices)), \
        list(map(lambda index: train_labels[index], random_indices))

def classify_tree(tree, val_feature):
    if type(tree) is int:
        return tree
    (feature, threshold), left, right = tree
    if val_feature[feature] < threshold:
        return classify_tree(left, val_feature)
    return classify_tree(right, val_feature)

def classify(val_feature, train_features, train_labels, forest):
    return list(map(lambda tree: classify_tree(tree, val_feature), forest))

def random_forest(T, train_features, train_labels, val_features):
    forest = []
    for i in range(T):
        bagging_train_features, bagging_train_labels = bagging(train_features, train_labels)
        forest.append(build_decision_tree(bagging_train_features, bagging_train_labels))
    labels = []
    for val_feature in val_features:
        label_list = classify(val_feature, train_features, train_labels, forest)
        vote, i = [0, 0], 0
        for label in label_list:
            vote[label] += 1
            i += 1
        if vote[0] >= vote[1]:
            labels.append(0)
        else:
            labels.append(1)
    return labels

def choose(T, train_features, train_labels, val_features, val_labels):
    result = random_forest(T, train_features, train_labels, val_features)
    i, error = 0, 0
    for label in val_labels:
        if label != result[i]:
            error += 1
        i += 1
    print("T = " + str(T) + ", error rate = " + str(error / i))
    file_name = "emailOutput" + str(T) + ".csv"
    with open(file_name, "w") as temp_file:
        temp_file.write("\n".join(map(lambda label: str(label), result)))

def main():
    with open("trainFeatures.csv") as temp_file:
        lines = temp_file.read().split("\n")
        train_features = [list(float(num) for num in line.split(',') if num) for line in lines]
    with open("trainLabels.csv") as temp_file:
        lines = temp_file.read().split("\n")
        train_labels = [int(line) for line in lines]
    with open("valFeatures.csv") as temp_file:
        lines = temp_file.read().split("\n")
        val_features = [list(float(num) for num in line.split(',') if num) for line in lines]
    with open("valLabels.csv") as temp_file:
        lines = temp_file.read().split("\n")
        val_labels = [int(line) for line in lines]
    choose(1, train_features, train_labels, val_features, val_labels)
    choose(2, train_features, train_labels, val_features, val_labels)
    choose(5, train_features, train_labels, val_features, val_labels)
    choose(10, train_features, train_labels, val_features, val_labels)
    choose(25, train_features, train_labels, val_features, val_labels)
    print("The best T is 10 or 25")
    print("Use T = 25 on test set")
    with open("testFeatures.csv") as temp_file:
        lines = temp_file.read().split("\n")
        test_features = [list(float(num) for num in line.split(',') if num) for line in lines]
    result = random_forest(25, train_features, train_labels, test_features)
    with open("emailOutput.csv", "w") as temp_file:
        temp_file.write("\n".join(map(lambda label: str(label), result)))

main()
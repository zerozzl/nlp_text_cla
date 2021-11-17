from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(golden_lists, predict_lists):
    acc = accuracy_score(golden_lists, predict_lists)
    pre = precision_score(golden_lists, predict_lists, average='micro')
    rec = recall_score(golden_lists, predict_lists, average='micro')
    f1 = f1_score(golden_lists, predict_lists, average='micro')
    return acc, pre, rec, f1


if __name__ == '__main__':
    gold = [1, 2, 0, 0]
    pred = [1, 1, 1, 1]

    acc, pre, rec, f1 = evaluate(gold, pred)
    print(acc, pre, rec, f1)

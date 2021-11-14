import pandas as pd
import numpy as np

def kNN (k):
    data = pd.read_excel("Asssignment4_data.xlsx").drop('House ID', axis=1)
    test = pd.read_excel("Asssignment4_data.xlsx", sheet_name=1).drop('House ID', axis=1)
    numpData = data.to_numpy()
    numpTest = test.to_numpy()

    testClassifications = np.empty(len(numpTest), dtype=object)

    for i in range(len(testClassifications)):
        distances = np.zeros(len(numpData))
        for j in range(len(distances)):
            distances[j] = np.linalg.norm(np.delete(numpData[j], -1) - np.delete(numpTest[i], -1))

        nearestNeighbors = distances.argsort()[:k]

        aptCnt = 0
        condoCnt = 0
        houseCnt = 0
        for j in range(len(nearestNeighbors)):
            if (numpData[nearestNeighbors[j], -1] == 'Apartment'):
                aptCnt = aptCnt + 1
            elif (numpData[nearestNeighbors[j], -1] == 'Condo'):
                condoCnt = condoCnt + 1
            else:
                houseCnt = houseCnt + 1

        print("\napt: ", aptCnt, " | condoCnt: ", condoCnt, " | houseCnt: ", houseCnt)

        if (aptCnt >= condoCnt and aptCnt >= houseCnt):
            testClassifications[i] = 'Apartment'
        elif (condoCnt >= aptCnt and condoCnt >= houseCnt):
            testClassifications[i] = 'Condo'
        else:
            testClassifications[i] = 'House'


    kAccuracy = accuracy(testClassifications, numpTest[:,-1])
    print("\nAccuracy: ", kAccuracy)

def accuracy(predicted, given):
    right = 0
    print("\npredicted:\n",predicted)
    print("\ngiven:\n",given)
    for i in range(len(predicted)):
        if (predicted[i] == given[i]):
            right = right + 1

    return right / len(predicted)

def main():
    kNN(10)


if __name__ == "__main__":
    main()


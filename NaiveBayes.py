import pandas as pd

"""
Conditional Probability_______________________________
| P(C_k AND X) = P(C_k | X) P(X) = P(X | C_k) P(C_k) |
|                                                    |
| Therfore:                                          |
|                                                    |
| P(C_k | X) = [P(X | C_k) * P(C_k)] / P(X)          |
|                                                    |
| With Naive Bayes, all features are independent,    |
| we can simplify the above equation to:             |
|                                                    |
|   P(C_k | X) = P(X | C_k) / P(C_k)
|____________________________________________________|

Target variable = Contruction Type

P(Apt | B ^ C ^ D ^ E ^ F ^ G ^ H ^ I) =

    P(B ^ C ^ D ^ E ^ F ^ G ^ H ^ I | apt) * P(apt)
    -----------------------------------------------
           P(B ^ C ^ D ^ E ^ F ^ G ^ H ^ I)

Given that Naive Bayes assumes all features are independant of
each other, we can simplify:

P(Apt | B ^ C ^ D ^ E ^ F ^ G ^ H ^ I) =
P(B | apt) * P(C | apt) * P(D | apt) * P(E | apt) * P(F | apt) * P(G | apt) * P(H | apt) * P(I | apt)
    -----------------------------------------------------------------------------------------------------
                        P(B) * P(C) * P(D) * P(E) * P(F) * P(G) * P(H) * P(I)

Computing conditional probabilities of Construction type, given feature:

APARTMENTS
-------------------------------------------------------------------------------------------------------------------------
P(Apartment) = 7/20

P(Local Price | Apartment) = (Total Local Price w/ Apartment + 1) / (Total of All Feautres /w Apartment + # of Features)
                           = (4.92 + 4.56 + 5.06 + 14.46 + 5.05 + 8.25 + 9.04 + 1) / (465.0912 + 8)
                           = 52.3292 / 473.0921 = 0.11061

P(Bathrooms   | Apartment) = (Total Bathrooms w/ Apartment + 1) / (Total of All Feautres /w Apartment + # of Features)
                           = (1 + 1 + 1 + 2.50 + 1 + 1.50 + 1 + 1) / (465.0912 + 8)
                           = 10 / 473.0921 = 0.02114

P(Land Area   | Apartment) = 43.7270 / 473.0921 =
P(Living area | Apartment) = 11.5350 / 473.0921 =
P(# Garages   | Apartment) = 9.5 / 473.0921     = 
P(# Rooms     | Apartment) = 49 / 473.0921      = 
P(# Bedrooms  | Apartment) = 25 / 473.0921      = 
P(Age of home | Apartment) = 272 / 473.0921     = 
--------------------------------------------------------------------------------------------------------------------------

CONDOS
--------------------------------------------------------------------------------------------------------------------------
P(Condo) = 6/20

P(Local Price | Condo) = 45.4954 / 412.9634  = 0.11017
P(Bathrooms   | Condo) = 9.0000 / 412.9634   = 0.02179
P(Land Area   | Condo) = 37.1480 / 412.9634  = 0.08995
P(Living area | Condo) = 10.3200 / 412.9634  = 0.02422
P(# Garages   | Condo) = 9.0000 / 412.9634   = 0.02179
P(# Rooms     | Condo) = 42.0000 / 412.9634  = 0.10170
P(# Bedrooms  | Condo) = 20.0000 / 412.9634  = 0.04843
P(Age of home | Condo) = 239.0000 / 412.9634 = 0.5787
--------------------------------------------------------------------------------------------------------------------------

HOUSE
--------------------------------------------------------------------------------------------------------------------------
P(Condo) = 7/20

P(Local Price | House) = 41.3252 / 423.4835 = 0.09758
P(Bathrooms   | House) = 8.50000 / 423.4835 = 0.02007
P(Land Area   | House) = 47.4163 / 423.4835 = 0.11198
P(Living area | House) = 10.7420 / 423.4835 = 0.02537
P(# Garages   | House) = 8.50000 / 423.4835 = 0.02007
P(# Rooms     | House) = 44.0000 / 423.4835 = 0.10390
P(# Bedrooms  | House) = 22.0000 / 423.4835 = 0.05195
P(Age of home | House) = 240.000 / 423.4835 = 0.56673
--------------------------------------------------------------------------------------------------------------------------


"""

def naiveBayesAlg():
    data = pd.read_excel("Asssignment4_data.xlsx").drop('House ID', axis=1)
    test = pd.read_excel("Asssignment4_data.xlsx", sheet_name=1).drop('House ID', axis=1)

    apartment = data[data['Construction type'].isin(['Apartment'])]
    condo = data[data['Construction type'].isin(['Condo'])]
    house = data[data['Construction type'].isin(['House'])]

    sumData = (data.drop('Construction type', axis=1)).sum()
    sumApt = (apartment.drop('Construction type', axis=1)).sum()
    sumCondo = (condo.drop('Construction type', axis=1)).sum()
    sumHouse = (house.drop('Construction type', axis=1)).sum()

    totalSumData = sumData.sum()
    totalSumApt = sumApt.sum()
    totalSumCondo = sumCondo.sum()
    totalSumHouse = sumHouse.sum()

    probApt = len(apartment) / len(data)
    probCondo = len(condo) / len(data)
    probHouse = len(house) / len(data)

    print("\n probApt, probCondo, probHouse")
    print(probApt, " ", probCondo, " ", probHouse)

    probDataFeatures = (sumData+1) / (totalSumData+8)
    condProbAptFeatures = (sumApt+1) / (totalSumApt+8)
    condProbCondoFeatures = (sumCondo+1) / (totalSumCondo+8)
    condProbHouseFeatures = (sumHouse+1) / (totalSumHouse+8)

    multCondProbAptFeat = condProbAptFeatures.prod()

    # Need to multiply condProbAptFeatures together and probDataFeatures together, then do equation below
    condProbApt = (condProbAptFeatures.prod() * probApt) / probDataFeatures.prod()
    condProbCondo = (condProbCondoFeatures.prod() * probCondo) / probDataFeatures.prod()
    condProbHouse = (condProbHouseFeatures.prod() * probHouse) / probDataFeatures.prod()
    print("\ncondProbApt")
    print(condProbApt)
    print("\ncondProbCondo")
    print(condProbCondo)
    print("\ncondProbHouse")
    print(condProbHouse)

    




    #################TESTING####################
    testApartment = data[data['Construction type'].isin(['Apartment'])]
    testCondo = data[data['Construction type'].isin(['Condo'])]
    testHouse = data[data['Construction type'].isin(['House'])]

    sumTest = (test.drop('Construction type', axis=1)).sum()
    sumTestApt = (testApartment.drop('Construction type', axis=1)).sum()
    sumTestCondo = (testCondo.drop('Construction type', axis=1)).sum()
    sumTestHouse = (testHouse.drop('Construction type', axis=1)).sum()

    totalTest = sumTest.sum()
    totalTestApt = sumTestApt.sum()
    totalTestCondo = sumTestCondo.sum()
    totalTestHouse = sumTestHouse.sum()

    # P(Apt | all X) = P(X | Apt) / P(X)

    print("\nsumTest")
    print(sumTest)
    print("\ntotalTest")
    print(totalTest)
    print("\nsumTestApt")
    print(sumTestApt)
    print("\ntotalTestApt")
    print(totalTestApt)

    condProbApt = (sumTestApt+1) / (totalTestApt+8)
    print("\ncondProbApt")
    print(condProbApt)

def main():
    naiveBayesAlg()


if __name__ == "__main__":
    main()

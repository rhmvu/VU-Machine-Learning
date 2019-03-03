import csv
import tensorflow as ts
import numpy as np

datafolder = "./data/"


def main():
    csvfile = open(datafolder+"train.csv", "r")
    trainreader = csv.reader(csvfile, delimiter=",", quotechar="\"")
    for row in trainreader:
        print(row)
    # TODO: Maybe we need to convert columns to int type if applicable and perform PCA? Then run algorithms on these top features?
    csvfile.close()


if __name__ == "__main__":
    main()
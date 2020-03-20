from csv import reader
from math import sqrt

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file, delimiter=' ')
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float - feature values
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to  int - class names
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
        print('[%s] => %d' % (value, i))
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1): # ignore last column - class name
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


def main():
    # Make a prediction with KNN
    dataset = load_csv('training1.txt')

    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)

    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0]) - 1)

    print("Load training data")
    #print(dataset)

    # define k value (# nearest neighbors)
    num_neighbors = 3

    print("Load test data")
    testData = load_csv('test.txt')
    for i in range(len(testData[0])):
        str_column_to_float(testData, i)

    for row in testData:
        # predict the label
        label = predict_classification(dataset, row, num_neighbors)
        print('Data=%s, Predicted: %s' % (row, label))


if __name__ == '__main__':
    main()

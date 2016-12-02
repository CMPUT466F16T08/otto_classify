import csv

'''
usage: read probabilities from csv file and make choice (choose the class with the largest probability)
input: file name (eg: prob.csv) 
output: a list containing chosen class (0, 1, 2, ...) [*start from 0]
'''
def read_prob_choose_class(filename):
    f = open(filename, 'rb')
    fr = csv.reader(f)
    array = [row[1:] for row in fr]
    f.close() 
    array = array[1:]
    
    choice = []
    for i in range(len(array)):
        line = array[i]
        max_ind = 0
        for j in range(len(line)):
            if float(line[j]) > float(line[max_ind]):
                max_ind = j
        choice.append(max_ind)
    return choice

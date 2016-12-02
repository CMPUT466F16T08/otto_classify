'''
usage: write results of prediction into a csv file
input: probs - probabilities for each class 
                   (a nx9 list: [[p0, p1, ..., p8] <- the 1st instance
                                 [p0,..........p8] <- the 2nd instance
                                 .........        
                                                  ])
output: a csv file called prob.csv
'''
def write_pred_prob(probs):
    f = open('prob.csv', 'wb')
    writer = csv.writer(f)
    labels = []
    for i in range(9):
        labels.append('class '+str(i))
    writer.writerow(labels)
    data = []
    for l in probs:
        data.append(l)
    writer.writerows(data)
    f.close()
    print 'finish writting <prob.csv>'
    
    

'''
usage: calculate log loss (total & each class)
input: actual - true labels (a one dimension list)
       predicted - probabilities for each class 
                   (a nx9 list: [[p0, p1, ..., p8] <- the 1st instance
                                 [p0,..........p8] <- the 2nd instance
                                 .........        
                                                  ])
       eps - there is no need to give this variable when you call this function
output: logloss for the WHOLE test set, 
        a list containing log loss for EACH class (format:[[class0, loglossForClass0],[class1,loglossForClass1],...])
'''

def log_loss_implement(actual, predicted, eps = 1e-15):
    predicted = np.minimum(np.maximum(predicted,eps),1-eps)  
    sum1 = 0
    N = len(actual)
    M = num_labels(actual)
    
    result_list = []
    
    for j in range(M):
        sum2 = 0
        count = 0
        for i in range(N):
            y = 1 if j==actual[i] else 0
            if j==actual[i]:
                y = 1
                count += 1
            else:
                y = 0
            p = predicted[i][j]
            temp = y*log(p)
            sum2 += temp
        cla_logloss = (-1)*sum2/float(count)
        print 'Class', j, 'log loss =', cla_logloss
        result_list.append([j, cla_logloss])
        
        sum1 += sum2
    logloss = (-1)*sum1/float(N)
    return logloss, result_list

'''
this function is called by log_loss_implement
'''
def num_labels(actual):
    labels = {}
    size = 0
    for l in actual:
        if l not in labels:
            size += 1
            labels[l] = 0
    return size
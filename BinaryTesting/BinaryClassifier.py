from qiskit.quantum_info import SparsePauliOp,Statevector
import numpy as np
def exactProbabilities(classifier,encoder,theta,train_x,test_x):
    train_P = []
    for x in train_x:
        tempqc = encoder.assign_parameters(x).compose(classifier.assign_parameters(theta))
        exp = Statevector(tempqc).expectation_value(SparsePauliOp('Z'*encoder.num_qubits)).real
        p_plus = (1+exp)/2
        p_minus = (1-exp)/2
        train_P.append({-1:p_minus,1:p_plus})
    
    test_P = []
    for x in test_x:
        tempqc = encoder.assign_parameters(x).compose(classifier.assign_parameters(theta))
        exp = Statevector(tempqc).expectation_value(SparsePauliOp('Z'*encoder.num_qubits)).real
        p_plus = (1+exp)/2
        p_minus = (1-exp)/2
        test_P.append({-1:p_minus,1:p_plus})
    return train_P, test_P

def exactLosses(classifier,encoder,theta,train_x,test_x,train_y,test_y):
    train_P, test_P = exactProbabilities(classifier=classifier,
                                    encoder=encoder,
                                    theta=theta,
                                    train_x=train_x,
                                    test_x=test_x)
    train_loss=0
    for i,x in enumerate(train_x):
        y=train_y[i]
        p_correct = train_P[i][y]
        p_incorrect = train_P[i][-y]
        train_loss+= ((p_correct-1)**2+p_incorrect**2)
        
    test_loss=0
    for i,x in enumerate(test_x):
        y=test_y[i]
        p_correct = train_P[i][y]
        p_incorrect = train_P[i][-y]
        test_loss+= ((p_correct-1)**2+p_incorrect**2)
    return train_loss,test_loss


def predictions(classifier,encoder,theta,train_x,test_x):
    
    train_P, test_P = exactProbabilities(classifier=classifier,
                                    encoder=encoder,
                                    theta=theta,
                                    train_x=train_x,
                                    test_x=test_x)

    test_pred = []
    for i,x in enumerate(test_x):
        if(test_P[i][-1]>0.5):
            test_pred.append(-1)
        else:
            test_pred.append(1)

    train_pred = []
    for i,x in enumerate(train_x):
        if(train_P[i][-1]>0.5):
            train_pred.append(-1)
        else:
            train_pred.append(1)

    pred_test_plus = [x for i,x in enumerate(test_x) if test_pred[i]==1]
    pred_test_minus = [x for i,x in enumerate(test_x) if test_pred[i]==-1]
    pred_train_plus = [x for i,x in enumerate(train_x) if train_pred[i]==1]
    pred_train_minus = [x for i,x in enumerate(train_x) if train_pred[i]==-1]
    
    return pred_test_plus,pred_test_minus,pred_train_plus,pred_train_minus

def accuracy(classifier,encoder,theta,train_x,test_x,train_y,test_y):
    train_P, test_P = exactProbabilities(classifier=classifier,
                                    encoder=encoder,
                                    theta=theta,
                                    train_x=train_x,
                                    test_x=test_x)

    train_total = 0
    for i,x in enumerate(train_x):
        y=train_y[i]
        p = train_P[i][y]
        if(p>0.5):
            train_total+=1
    
    test_total = 0
    for i,x in enumerate(test_x):
        y=test_y[i]
        p = test_P[i][y]
        if(p>0.5):
            test_total+=1
    return train_total/len(train_x),test_total/len(test_x)
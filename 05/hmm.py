#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[10]:


def forward_algorithm(O, HMM_model):                #前向算法 O是观测序列
    """HMM Forward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment

    # Put Your Code Here
    alpha = [[0 for i in range(T)] for j in range(N)]
    for i in range(N):
        alpha[i][0] = pi[i]
    for i in range(1,T):
        for j in range(N):
            for r in range(N):
                alpha[j][i] += alpha[r][i-1]*A[r][j]*B[j][O[i]]
    for i in range(T):
        prob += alpha[i][T-1]

    # End Assignment
    return prob


# In[12]:


def backward_algorithm(O, HMM_model):                #后向算法
    """HMM Backward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment

    # Put Your Code Here
    beta = [[0 for i in range(T)] for j in range(N)]
    for i in range(N):
        beta[i][T-1] = 1
    for i in range(T-2,-1,-1):
        for j in range(N):
            for r in range(N):
                beta[j][i] += beta[r][i+1]*A[j][r]*B[r][O[i+1]]
    for i in range(T):
        prob += beta[i][0]*B[i][O[0]]*pi[i]

    # End Assignment
    return prob


# In[29]:


def Viterbi_algorithm(O, HMM_model):                #维特比算法
    """Viterbi decoding.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Returns:
        best_prob: the probability of the best state sequence
        best_path: the best state sequence
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    best_prob, best_path = 0.0, []
    # Begin Assignment

    # Put Your Code Here
    delta = [[0 for i in range(N)] for j in range(T)]
    phi = [[0 for i in range(N)] for j in range(T)]
    for i in range(N):
        delta[i][0] = pi[i]*B[i][O[0]]
    for i in range(1,T):
        for j in range(N):
            maxdelta = 0
            path = 0
            for r in range(N):
                temp = delta[r][i-1]*A[r][j]
                if temp>maxdelta:
                    maxdelta = temp
                    path = r
            delta[j][i] = maxdelta*B[j][O[i]]
            phi[j][i] = path
    
    for i in range(N):
        if best_prob<delta[T-1][i]:
            best_prob = delta[T-1][i]
            lastpath = i
    lastpath += 1
    best_path = [lastpath]
    #print(phi,delta)
    for i in range(T-1,-1,-1):
        lastpath = phi[lastpath-1][i]
        best_path.insert(0,lastpath)
        
    # End Assignment
    return best_prob, best_path


# In[30]:


if __name__ == "__main__":
    color2id = { "RED": 0, "WHITE": 1 }
    # model parameters
    pi = [0.2, 0.4, 0.4]
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    # input
    observations = (0, 1, 0)
    HMM_model = (pi, A, B)
    # process
    observ_prob_forward = forward_algorithm(observations, HMM_model)
    print(observ_prob_forward)

    observ_prob_backward = backward_algorithm(observations, HMM_model)
    print(observ_prob_backward)

    best_prob, best_path = Viterbi_algorithm(observations, HMM_model) 
    print(best_prob, best_path)


# In[ ]:





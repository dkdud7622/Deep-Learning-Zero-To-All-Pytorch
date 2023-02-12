import numpy as np
import torch

'''
numpy version
'''
print("______numpy______")
t = np.array([0.,1.,2.,3.,4.,5.,6.])
print(t)

print("Rank of t: ", t.ndim) # 몇개의 텐서니? 벡터니 행렬이니 텐서니? 
print("Shape of t: ", t.shape)

print('t[0] t[1] t[-1] = ',t[0], t[1], t[-1], )
print('t[2:5] t[4:-1] = ',t[2:5],  t[4:-1], )
print('t[:2] t[3:] = ',t[:2], t[3:] )


t2 = np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.,],[10.,11.,12.]])
print(t2)
print()

print("Rank of t: ", t2.ndim)
print("Shape of t: ", t2.shape)
print("______________________")
print()
print()


'''
pytorch version
'''
print("______pytorch______")
t = torch.FloatTensor([0.,1.,2.,3.,4.,5.,6.])
print(t)

print(t.dim())  #rank
print(t.shape)  #shape
print(t.size()) #shape
print(t[0],t[1],t[-1])
print(t[2:5],t[4:-1])
print(t[:2],t[3:])
print()
t2 = torch.FloatTensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.,],[10.,11.,12.]])
print(t2)

print(t2.dim())  #rank
print(t2.size()) #shape
print(t2[:,1])
print(t2[:,1].size())
print(t2[:,:-1])
print("______________________")
print()
print()


'''
 Broadcasting : 다른 크기의 행렬(벡터,텐서) 연산시 자동으로 사이즈를 맞춰줌
'''
print("______Broadcasting______")
# Same shape (1,2)
m1 = torch.FloatTensor([[3,2]])
m2 = torch.FloatTensor([[2,2]])
print(m1 + m2)

# Vector(1,2) + scalar
m1 = torch.FloatTensor([[1,2]])
m2 = torch.FloatTensor([3]) # 3-> [[3, 3]]
print(m1 + m2)

# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([[1,2]])
m2 = torch.FloatTensor([[3],[4]])
print(m1 + m2)
print("______________________")
print()
print()

'''
Multiplication vs Matrix Multiplication
'''
print("______Mul______")
print('_____________')
print("Mul vs Matmul")
print('_____________')
m1 = torch.FloatTensor([[1,2],[3,4]])
m2 = torch.FloatTensor([[1],[2]])
print(m1.matmul(m2))
print(m1 * m2)
print(m1.mul(m2))
print("______________________")
print()
print()


'''
Mean
'''
print("______Mean______")
t = torch.FloatTensor([1,2])
print(t.mean())
print()
# Can't use mean() on integera
# t = torch.LongTensor([1,2])
# try :
#     print(t.mean())
# except  Exception as exc :
#     print(exc)

t = torch.FloatTensor([[1,2],[3,4]])
print(t)
print()
print(t.mean())     # 전체에 대한 평균
print(t.mean(dim=0))# → 방향이 없어져야함 ( 2x2 => 1x2 ) 따라서 (1+3)/2 , (2+4)/2 -=> 2,3
print(t.mean(dim=1))# ↓ 방향이 없어져야함 따라서 (1+2)/2 , (3+4)/2 => [1.5 , 3.5]T
print(t.mean(dim=-1))
print("______________________")
print()
print()

'''
Sum
'''
print("______SUM______")
t = torch.FloatTensor([[1,2],[3,4]])
print(t)

print(t.sum())
print(t.sum(dim=0))
print(t.sum(dim=1))
print(t.sum(dim=-1))
print("______________________")
print()
print()

'''
Max and Argmax
'''
print("______Max and Argmax______")
t = torch.FloatTensor([[1,2],[3,4]])
print(t)
print()
print(t.max())
print()
#각 col중 큰것만 남기고 1x2 행렬이 됨. => [3., 4. ]
print(t.max(dim=0))  #[0]:max | [1]:Argmax  
print("Max: ", t.max(dim=0)[0])
print("Argmax: ", t.max(dim=0)[1])
print()
print(t.max(dim=1))
print()
print(t.max(dim=-1))
print("______________________")
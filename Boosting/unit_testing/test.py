import multiprocessing
import numpy as np

matrix = np.ones((100,100))

def square(x):
    ans =0
    global matrix
    for i in range(1,50):
        ans += matrix[x[0]][x[1]]
        print(ans)
    return ans
        

mylist = list(range(0,1000))

pool = multiprocessing.Pool()

result = [pool.apply_async(square,[idx,]) for idx, x in np.ndenumerate(matrix)]
output = [p.get() for p in result]
print(output)

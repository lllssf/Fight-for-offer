# ThoughtWorks编程题解答 将数字转换为lcd的表现形式
import sys
lcd = [['._.','|.|','|_|'],
       ['...','..|','..|'],
       ['._.','._|','|_.'],
       ['._.','._|','._|'],
       ['...','|_|','..|'],
       ['._.','|_.','._|'],
       ['._.','|_.','|_|'],
       ['._.','..|','..|'],
       ['._.','|_|','|_|'],
       ['._.','|_|','..|']]
array = []
for line in sys.stdin:
    a=len(line)-1
    number=int(line)
    for i in range(a):
        b = int(number/(10**(a-1-i)))
        number -= b*(10**(a-1-i))
        array.append(lcd[b])
    break
for i in range(3):
    for j in range(a-1):
        print(array[j][i],end=' ')
    print(array[a-1][i])

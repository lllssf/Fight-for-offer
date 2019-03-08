import sys
lines = sys.stdin.readlines()
work = {}
for line in lines[1:-1]:
    if not line.strip().split():
        continue
    [d,p] = map(int, line.strip().split())
    work[d] = max(work.get(d,0),p)
#按工作难度排序
sortWork = sorted(work.items(),key = lambda x:x[0])
#每个人能力排序
A = enumerate(map(int,lines[-1].strip().split()))
A = sorted(A,key = lambda x:x[1])
maxP = 0
people = 0
M = len(A)
N = len(work)
pay = [0]*M
for i in range(N):
    while sortWork[i][0]>A[people][1]:
        pay[A[people][0]] = maxP
        people += 1
        if people >= M:
            break
    if people >= M:
            break
    if maxP<sortWork[i][1]:
        maxP = sortWork[i][1]
#多个人能力高于最高工作难度
for i in range(people,M):
    pay[A[i][0]] = maxP
for i in pay:
    print(i)

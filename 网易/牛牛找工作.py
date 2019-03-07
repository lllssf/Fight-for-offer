import sys
lines = sys.stdin.readlines()
work = {}
for line in lines[1:-1]:
    if not lines.strip().split():
        continue
    [d,p] = map(int,line().strip().split())
    work[d] = max(work.get(d,0),b)
A = list(enumerate(map(int,lines[-1].strip().split())))
M = len(A)
N = len(work)
dwork = sorted(work.items(),key = lambda x:x[0])
sA = sorted(A,key = lambda x:x[1])
pay = [0] *M
for i in range(M):
    

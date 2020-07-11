import itertools

def gen_permutation(positions):
    res = []
    for i in range(len(positions)+1):
        for x in itertools.combinations(positions,i):
            tmp = []
            for o in x:
                tmp.append(o)
            res.append(tmp)
    return res


a = [[6,7],[8,8],[1,1]]


print(gen_permutation(a))




'''
# new_state.split("pos",1)[1])
s = 'x9y0pos67'

x = s.split("x",1)[1].split("y",1)[0]
y = s.split("y",1)[1].split("p",1)[0]
print(x)
print(y)
'''

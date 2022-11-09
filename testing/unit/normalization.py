from gym_flp.util import preprocessing

y=[]
for x in range(30):
    y.append(preprocessing.normalize(a=-1, b=1, x=x, x_max=30, x_min=0))

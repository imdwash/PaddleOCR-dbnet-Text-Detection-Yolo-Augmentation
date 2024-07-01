import random 

def rotate_aug_values(n,rotate_list):
    angle=[]
    for i in range(int(rotate_list[0]),int(rotate_list[1])+5,5):
        angle.append(i)
    random_angle=[]
    for i in range(0,n):
        random_angle.append(random.choice(angle))
    return random_angle
    
def blur_aug_values(n,blvalue_list):
    b_values=[]
    for i in range(int(blvalue_list[0]),int(blvalue_list[1])+3,3):
        b_values.append(i)
    random_b_values=[]
    for i in range(0,n):
        random_b_values.append(random.choice(b_values))
    return random_b_values

def brighten_aug_values(n,cvalue_list):
    c_values=[]
    for i in range(int(cvalue_list[0]),int(cvalue_list[1])+10,10):
        c_values.append(i)
    random_c_values=[]
    for i in range(0,n):
        random_c_values.append(random.choice(c_values))
    return random_c_values

def darken_aug_values(n,dvalue_list):
    d_values=[]
    for i in range(int(dvalue_list[0]),int(dvalue_list[1])+10,10):
        d_values.append(i)
    random_d_values=[]
    for i in range(0,n):
        random_d_values.append(random.choice(d_values))
    return random_d_values

def elastic_transform(n,evalue_list):
    e_values=[]
    for i in range(int(evalue_list[0]),int(evalue_list[1])+20,20):
        e_values.append(i)
    random_e_values=[]
    for i in range(0,n):
        random_e_values.append(random.choice(e_values))
    return random_e_values

def rigid(n,rvalue_list):
    r_values=[]
    for i in range(int(rvalue_list[0]),int(rvalue_list[1])+1,1):
        r_values.append(i)
    random_r_values=[]
    for i in range(0,n):
        random_r_values.append(random.choice(r_values))
    return random_r_values



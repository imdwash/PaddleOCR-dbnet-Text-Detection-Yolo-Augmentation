import aug
import os
import argparse
import ChangeNameDet
import DetSplit
import ImgTextSplit
import GenLabel
import AugmentValues
import makeyml
import yolo_operations
import random
import shutil

parser=argparse.ArgumentParser() #CLI documentation for commandline input
parser.add_argument('--type=',type=str)
parser.add_argument('--dir=',type=str)
parser.add_argument('--n=',type=str)
parser.add_argument('--rotate=',type=str)
parser.add_argument('--zrot=',type=str)
parser.add_argument('--blur=',type=str)
parser.add_argument('--brighten=',type=str)
parser.add_argument('--darken=',type=str)
parser.add_argument('--dis=',type=int)
parser.add_argument('--elastic=',type=str)
parser.add_argument('--rigid=',type=str)
parser.add_argument('--lbcor=',type=str)
parser.add_argument('--percent=',type=int)
args=parser.parse_args()

type=vars(args)['type=']
dir=vars(args)['dir=']
n=vars(args)['n=']
blvalue=vars(args)['blur=']
rotate=vars(args)['rotate=']
zrot=vars(args)['zrot=']
cvalue=vars(args)['brighten=']
dvalue=vars(args)['darken=']
dis=vars(args)['dis=']
percent=vars(args)['percent=']
evalue=vars(args)['elastic=']
rvalue=vars(args)['rigid=']
lbcord=vars(args)['lbcor=']

dir_name = dir+'Detection_Dataset'
n_of_images=int(len(os.listdir(dir))/2)

if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
    print('Detection_Dataset Folder Created')
outdir=dir_name+'/'
count=0
n=int(n.replace('x',''))
rotate_list=AugmentValues.rotate_aug_values(n,rotate.split('to'))
blvalue_list=AugmentValues.blur_aug_values(n,blvalue.split('to'))
cvalue_list=AugmentValues.brighten_aug_values(n,cvalue.split('to'))
dvalue_list=AugmentValues.darken_aug_values(n,dvalue.split('to'))
evalue_list=AugmentValues.elastic_transform(n,evalue.split('to'))
rvalue_list=AugmentValues.rigid(n,rvalue.split('to'))

k=0
t=0

if type=='yolo':

    if not os.path.isdir(dir+'withcrop'):
        os.mkdir(dir+'withcrop')
    else:
        pass
    for filename in os.listdir(dir):
        if filename.endswith('.jpg') or filename.endswith('.txt'):
            shutil.copy(dir+'/'+filename,dir+'withcrop/')
        else:
            pass
            

    dir=dir+'withcrop'

    img_list=os.listdir(dir)
    crop_len=int(len(img_list)/3)
    random_img_list=random.sample(img_list, len(img_list))

    print('Croping for Yolo Started')
    for filename in random_img_list:
        if '.jpg' in filename:
            
            if os.path.exists(os.path.join(dir,filename.replace('.jpg','.txt'))):
                os.chdir(dir)
                g=open(filename.replace('.jpg','.txt'),'r')
                
                if len(g.readlines())!=0:
                
                    if t<crop_len:
                        t=yolo_operations.y_crop(filename,dir+'/',n,dis,crop_len,t)
                    else:
                        pass
                else:
                    pass
                g.close()
            else:
                pass

    print('Croping for Yolo Ended')     

print("Augmentation Started")

for filename in os.listdir(dir): #Applying Augmentaion
    if '.jpg' in filename:
        os.chdir(dir)
        if os.path.exists(os.path.join(dir,filename.replace('.jpg','.txt'))):
            pass
        else:
            g=open(filename.replace('.jpg','.txt'),'w')
            g.write("")
            g.close()
        
        j=0
        k=[]

        k.append(1)
        for _ in range(0,n-1):
            if type=='yolo':
                k.append(random.randint(2,8))
            else:
                k.append(random.randint(2,7))
        
        for i in range(0,n):
            
            if j<n and k[i]==1:
                aug.original(filename,outdir) #original
                count+=1
                j+=1
                if count%100==0:
                    print("Number of data created: ",count)
                    
            if j<n and k[i]==2:
                aug.rotate(filename,rotate_list[i],outdir,type,zrot) #rotation
                count+=1
                j+=1
                if count%100==0:
                    print("Number of data created: ",count)
                    
            if j<n and k[i]==3:
                aug.blur(filename,blvalue_list[i],outdir)#bluring
                count+=1
                j+=1
                if count%100==0:
                    print("Number of data created: ",count)
            
            if j<n and k[i]==4:
                aug.darken(filename,dvalue_list[i],outdir) #darken
                j+=1
                count+=1
                if count%100==0:
                    print("Number of data created: ",count)
                
            if j<n and k[i]==5:
                aug.brighten(filename,cvalue_list[i],outdir) #brighten
                j+=1
                count+=1
                if count%100==0:
                    print("Number of data created: ",count)
                
            if j<n and k[i]==6:
                aug.elastic_transform(filename,evalue_list[i],outdir) #elastic
                j+=1
                count+=1
                if count%100==0:
                    print("Number of data created: ",count)
                
            if j<n and k[i]==7:
                aug.rigid(filename,rvalue_list[i],outdir) #elastic
                j+=1
                count+=1
                if count%100==0:
                    print("Number of data created: ",count)
            
            if j<n and k[i]==8:
                aug.flip(filename,outdir,random.randint(0,1)) #random flip
                j+=1
                count+=1
                if count%100==0:
                    print("Number of data created: ",count)
        
print("Total number of data created: ",count)

print("Augmentation Completed")

if type=='yolo' and lbcord!='no':
    
    if not os.path.isdir(outdir+'letterboxoutput/'):
        os.mkdir(outdir+'letterboxoutput/')
    else:
        pass
    dir=outdir
    outdir=dir+'letterboxoutput/'
    os.chdir(dir)
    print('LetterBox Started')
    yolo_operations.letterbox(dir,outdir,lbcord)
    print('LetterBox Ended')
     
if type=='paddle':
    print("Name Changing Started")
    ChangeNameDet.change_name(outdir) #change name of image and annotation
    print("Name Changing Completed")

print("Splitting Started")
DetSplit.split(outdir,percent)#split into train and test

ImgTextSplit.img_text_split(outdir,type)#split image and annotation
print("Complete Splitting")

if type=='yolo':
    makeyml.makeYML(dir,outdir)
    print("YAML file saved in ",dir)

if type=='paddle':
    print('Gen Label Started')
    GenLabel.gen_det_label(outdir+'Train/',outdir+'TrainAnno',outdir+'train_label.txt') #apply genlabel
    GenLabel.gen_det_label(outdir+'Test/',outdir+'TestAnno',outdir+'test_label.txt')
    print('Gen label completed')

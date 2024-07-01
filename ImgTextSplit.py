import shutil
import os
import yaml


def img_text_split(dir:str,type)->None:
    '''
    This function splits images and annotation in different folder for both train and test
    '''
    if type=='paddle':
        if 'TrainAnno' not in os.listdir(dir):
            os.makedirs(dir+'TrainAnno')

        for trname in os.listdir(dir+'Train'):
            if '.txt' in trname:
                shutil.move(dir+'Train/'+trname,dir+'TrainAnno/'+trname) 

        if 'TestAnno' not in os.listdir(dir):
            os.makedirs(dir+'TestAnno')

        for tename in os.listdir(dir+'Test'):
            if '.txt' in tename:
                shutil.move(dir+'Test/'+tename,dir+'TestAnno/'+tename)

    else:
        dir2=dir+'Train/'
        if 'images' not in os.listdir(dir2):
            os.makedirs(dir2+'images')
        if 'labels' not in os.listdir(dir2):
            os.makedirs(dir2+'labels')

        for trimg in os.listdir(dir2):
            if '.jpg' in trimg:
                shutil.move(dir+'Train/'+trimg,dir2+'images')
            if '.txt' in trimg:
                shutil.move(dir+'Train/'+trimg,dir2+'labels')
        
        dir2=dir+'Test/'
        if 'images' not in os.listdir(dir2):
            os.makedirs(dir2+'images')
        if 'labels' not in os.listdir(dir2):
            os.makedirs(dir2+'labels')

        for trimg in os.listdir(dir2):
            if '.jpg' in trimg:
                shutil.move(dir+'Test/'+trimg,dir2+'images')
            if '.txt' in trimg:
                shutil.move(dir+'Test/'+trimg,dir2+'labels')

        


   

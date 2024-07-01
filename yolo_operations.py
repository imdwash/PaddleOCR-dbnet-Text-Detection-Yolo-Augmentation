
from PIL import Image
import time
import numpy as np
import os
import random
import cv2
import pybboxes as pbx

def y_crop(dir,outdir,n,d,q,i):
    img=dir
    img2=Image.open(img)
    bb=img.replace('.jpg','.txt')
    
    g=open(bb,'r')
    
    l=len(g.readlines())
    g.close()
    
    for i in range(0,l):
        if i <q:
            f=open(bb,'r')
            x=(float(f.readlines()[i].split(" ")[1:][0]))
            f.close()

            f=open(bb,'r')
            y=(float(f.readlines()[i].split(" ")[1:][1]))
            f.close()

            f=open(bb,'r')
            w=(float(f.readlines()[i].split(" ")[1:][2]))
            f.close()

            f=open(bb,'r')
            h=(float(f.readlines()[i].split(" ")[1:][3]))
            f.close()
            image_w=img2.size[0]
            image_h=img2.size[1]

            w = w * image_w
            h = h * image_h
            x1 = ((2 * x * image_w) - w)/2
            y1 = ((2 * y * image_h) - h)/2
            x2 = x1 + w
            y2 = y1 + h

            xmin = round(x1)
            xmax = round(x2)
            ymin = round(y1)
            ymax = round(y2)
            
            img3=img2.crop((xmin,ymin,xmax,ymax))
            
            img4=img2.crop((xmin-d,ymin-d,xmax+d,ymax+d))
            x_center = (d + img3.size[0]-d + img3.size[0]-d+d) / 4
            y_center = (d +d +img3.size[1]-d + img3.size[1]-d) / 4

            # Calculate the width and height of the bounding box
            width = max(d , -d+img3.size[0] , -d+img3.size[0],d) - min(d , -d+img3.size[0] , -d+img3.size[0],d)
            height = max(d ,d , -d+img3.size[1] , -d+img3.size[1]) - min(d ,d , -d+img3.size[1] , -d+img3.size[1])

            # Normalize the coordinates, width, and height to be between 0 and 1
            try:
                x_center /= img3.size[0]
                y_center /= img3.size[1]
                width /= img3.size[0]
                height /= img3.size[1]
                c=[x_center,y_center,width,height]
                f=open(bb,'r')
                a=f.readlines()[i].split(' ')[0]
                f.close()
                img4.save(outdir+img.replace('.jpg','_crop_'+str(i)+'_'+str(n)+'.jpg'))
                g=open(outdir+bb.replace('.txt','_crop_'+str(i)+'_'+str(n)+'.txt'),'a')
                g.write(a+' '+str(c).replace(']','').replace('[','').replace(',',''))
                g.write('\n')
                g.close
            except ZeroDivisionError:
                pass

    return i
    

def letterbox(dir,outdir,shape):

    img_list=os.listdir(dir)

    random_img_list=random.sample(img_list, len(img_list)-1)


    print('Croping for Yolo Started')
    for filename in random_img_list:
        if '.jpg' in filename:
            img=dir+filename
            img2=Image.open(img)
            bb=img.replace('.jpg','.txt')
            txtname=filename.replace('.jpg','.txt')
            name=str(time.time())
            g=open(bb,'r')
            l=len(g.readlines())
            g.close()

            def preprocess(img2, input_shape,img_h,img_w):
                x_scale = 1
                y_scale = 1
                offset_h = 0
                offset_w = 0
            
                new_h, new_w = input_shape[1], input_shape[0] # desired input shape for the model
                offset_h, offset_w = 0, 0  # initialize the offset
                if (new_w / img_w) <= (new_h / img_h):
                    new_h = int(img_h * new_w / img_w)
                    offset_h = (input_shape[0] - new_h) // 2
                else:
                    new_w = int(img_w * new_h / img_h)
                    offset_w = (input_shape[1] - new_w) // 2
                resized=img2.resize((new_w,new_h), resample=Image.BILINEAR)
                img2 = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
                img2[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized

                # Calculate scaling factors for the bounding box coordinates
                x_scale = new_w / img_w
                y_scale = new_h / img_h
                    

                return img2, [x_scale, y_scale, offset_h, offset_w]
                    

            img_w, img_h = img2.size
            imgo,add_data=preprocess(img2,(int(shape.split(',')[0]),int(shape.split(',')[1])),img_h,img_w)
            imgo = cv2.cvtColor(imgo, cv2.COLOR_BGR2RGB)
            
            x_scale, y_scale, offset_h, offset_w = add_data[:]
            
            if l==0:
                    
                cv2.imwrite(outdir+str(name)+'.jpg', imgo)
                g=open(outdir+str(name)+'.txt','a')
                g.write('')
                g.close

            else:
        
                for i in range(0,l):
                    
                    f=open(bb,'r')
                    x=(float(f.readlines()[i].split(" ")[1:][0]))
                    f.close()

                    f=open(bb,'r')
                    y=(float(f.readlines()[i].split(" ")[1:][1]))
                    f.close()

                    f=open(bb,'r')
                    w=(float(f.readlines()[i].split(" ")[1:][2]))
                    f.close()

                    f=open(bb,'r')
                    h=(float(f.readlines()[i].split(" ")[1:][3]))
                    f.close()
                    normalized_bbox = [x, y, w, h]
                    bbox = pbx.convert_bbox(normalized_bbox, from_type="yolo", to_type="voc", image_size=(img_w,img_h))
                    preserved_bbox = [bbox[0] * x_scale + offset_w, bbox[1] * y_scale + offset_h, bbox[2] * x_scale + offset_w, bbox[3] * y_scale + offset_h]
                    c = pbx.convert_bbox(preserved_bbox, from_type="voc", to_type="yolo", image_size=(imgo.shape[1],imgo.shape[0]))

                    f=open(bb,'r')
                    a=f.readlines()[i].split(' ')[0]
                    f.close()
                    
                    cv2.imwrite(outdir+str(name)+'.jpg', imgo)
                    g=open(outdir+str(name)+'.txt','a')
                    g.write(a+' '+str(c).replace(']','').replace('[','').replace(',','').replace('(','').replace(')',''))
                    g.write('\n')
                    g.close

from PIL import Image, ImageOps
import os
import random
dir='C:/innate2/Batch_code_inspection_system/Automatic_Det_generateV2/images/'
def flip(img_name,bb,rand,j):
    image=Image.open(img_name)
    if rand==1:
        flipped_image =  ImageOps.mirror(image) #horizontal flip
    else:
        flipped_image = ImageOps.flip(image) #vertical
    g=open(bb,'r')
    l=len(g.readlines())
    g.close()
    flipped_image.save(str(j)+'.jpg')
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

        image_w=image.size[0]
        image_h=image.size[1]

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
        #print(xmin,xmax,ymin,ymax)
        def mirror_hor(points, image_width):
            mirrored_points = []
            for x, y in points:
                mirrored_x = image_width - x
                mirrored_points.append((mirrored_x, y))
            return mirrored_points
        
        def mirror_ver(points, image_height):
            mirrored_points = []
            for x, y in points:
                mirrored_y = image_height - y
                mirrored_points.append((x, mirrored_y))
            return mirrored_points
  
        if rand==1:
            b=mirror_hor([(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)],image_w)
        else:
            b=mirror_ver([(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)],image_h)

        x_center = (b[0][0] + b[1][0] + b[2][0] + b[3][0]) / 4
        y_center = (b[0][1] + b[1][1] + b[2][1] + b[3][1]) / 4

        # Calculate the width and height of the bounding box
        width = max(b[0][0] , b[1][0] , b[2][0] , b[3][0]) - min(b[0][0] , b[1][0] , b[2][0] , b[3][0])
        height = max(b[0][1] , b[1][1] , b[2][1] , b[3][1]) - min(b[0][1] , b[1][1] , b[2][1] , b[3][1])

        # Normalize the coordinates, width, and height to be between 0 and 1
        x_center /= image.size[0]
        y_center /= image.size[1]
        width /= image.size[0]
        height /= image.size[1]
        c=[x_center,y_center,width,height]
        f=open(bb,'r')
        a=f.readlines()[i].split(' ')[0]
        f.close()
        os.chdir(dir)
        
        
        g=open(str(j)+'.txt','a')
        g.write(a+' '+str(c).replace(']','').replace('[','').replace(',',''))
        g.write('\n')
        g.close
        
j=0
for filename in os.listdir(dir):

    if '.jpg' in filename:
        text=filename.replace('.jpg','.txt')
        flip(dir+filename,dir+text,random.randint(0,1),j)
    j+=1
    



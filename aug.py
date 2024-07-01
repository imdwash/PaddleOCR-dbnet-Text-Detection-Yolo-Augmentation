import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps,ImageDraw
import random
import shutil
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from random import randint,choice
import time

def rotate(dir,angle,outdir,type,zrot): 
    
    img2=Image.open(dir)
    #img.show()
    bb=dir.replace('.jpg','.txt')
    #angle=15
    f=open(bb,'r')
    l=len(f.readlines())
    f.close()
    name=str(time.time())
    def rotateBox(image,angle,bbox):
    # Load the image
        #image=cv2.resize(image)
        # Convert the image from RGB to BGR
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Get the image dimensions
        height, width = image.shape[:2]
        image_center = (width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        # Define the rotation angle
        value = angle

        # Compute the rotation matrix
        rotation_mat = cv2.getRotationMatrix2D(image_center, value, 1.)

        # Compute the new dimensions of the image after rotation
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # Update the rotation matrix with the new dimensions
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # Rotate the image using the rotation matrix
        rotated_image = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))

        # Define the bounding box coordinates
        bounding_box = bbox

        # Reshape the bounding box coordinates
        bounding_box = bounding_box.reshape((-1, 1, 2))

        # Rotate the bounding box coordinates using the rotation matrix
        rotated_bounding_box = cv2.transform(bounding_box, rotation_mat)

        coordinates_text = rotated_bounding_box.reshape((-1, 1, 2))
        coordinates_text=coordinates_text.flatten()
        rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb),coordinates_text
    if type=='paddle':
        bb_list=[]
        overall=[]
        if l==0:
                    
                img2.save(outdir+newdir)
                g=open(outdir+newbb,'a')
            
                g.write('')
                g.close

        else:
            for i in range(0,l):
                f=open(bb,'r')
                x1=f.readlines()[i].split(',')[0]
                f.close()

                f=open(bb,'r')
                y1=f.readlines()[i].split(',')[1]
                f.close()

                f=open(bb,'r')
                x2=f.readlines()[i].split(',')[2]
                f.close()

                f=open(bb,'r')
                y2=f.readlines()[i].split(',')[3]
                f.close()

                f=open(bb,'r')
                x3=f.readlines()[i].split(',')[4]
                f.close()

                f=open(bb,'r')
                y3=f.readlines()[i].split(',')[5]
                f.close()

                f=open(bb,'r')
                x4=f.readlines()[i].split(',')[6]
                f.close()

                f=open(bb,'r')
                y4=f.readlines()[i].split(',')[7]
                f.close()

                bobox = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                
                img,cor=rotateBox(img2,angle,bobox)
                
                imgo=img
                numpy_array = np.array(img)
                
                
                # Convert the NumPy array to OpenCV format
                img= cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
                bb_list.append([(int(cor[0]), int(cor[1])), (int(cor[2]), int(cor[3])), (int(cor[4]), int(cor[5])), (int(cor[6]), int(cor[7]))])
                
                for j in range(0,4):
                    overall.append(bb_list[i][j])
                sortoverall_x=sorted(overall,key=lambda sortoverall:sortoverall[0])
                sortoverall_y=sorted(overall,key=lambda sortoverall:sortoverall[1])
                
                if sortoverall_y[0][1]-(int(0.25*img.shape[1]))<0 and sortoverall_x[0][0]-(int(0.25*img.shape[0]))<0:
                
                    a=int(0.25*img.shape[1])+sortoverall_y[0][1]-(int(0.25*img.shape[1]))
                    b=int(0.25*img.shape[0])+sortoverall_x[0][0]-(int(0.25*img.shape[0]))
                    
                    img3=img[a:sortoverall_y[-1][1]+(int(0.25*img.shape[1])),b:sortoverall_x[-1][0]+(int(0.25*img.shape[0]))]

                elif sortoverall_y[0][1]-(int(0.25*img.shape[1]))<0 and sortoverall_x[0][0]-(int(0.25*img.shape[0]))!=0:
                    
                    a=int(0.25*img.shape[1])+sortoverall_y[0][1]-(int(0.25*img.shape[1]))
                    img3=img[a:sortoverall_y[-1][1]+(int(0.25*img.shape[1])),sortoverall_x[0][0]-(int(0.25*img.shape[0])):sortoverall_x[-1][0]+(int(0.25*img.shape[0]))]
                elif sortoverall_y[0][1]-(int(0.25*img.shape[1]))!=0 and sortoverall_x[0][0]-(int(0.25*img.shape[0]))<0:
                        b=int(0.25*img.shape[0])+sortoverall_x[0][0]-(int(0.25*img.shape[0]))
                        img3=img[sortoverall_y[0][1]-(int(0.25*img.shape[1])):sortoverall_y[-1][1]+(int(0.25*img.shape[1])),b:sortoverall_x[-1][0]+(int(0.25*img.shape[0]))]

                else:
                    img3=img[sortoverall_y[0][1]-(int(0.25*img.shape[1])):sortoverall_y[-1][1]+(int(0.25*img.shape[1])),sortoverall_x[0][0]-(int(0.25*img.shape[0])):sortoverall_x[-1][0]+(int(0.25*img.shape[0]))]
                
                img = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

                # Create a Pillow image from the RGB image
                img = Image.fromarray(img)
                

                f=open(bb,'r')
                a=f.readlines()[i].split(',')[8]
                f.close()
                newdir=dir.replace('.jpg','_r_'+name+'.jpg')
                newbb=bb.replace('.txt','_r_'+name+'.txt')
                if zrot =='yes':
                    if random.randint(0, 1)==0:
                    
                        img.save(outdir+newdir)

                    else:
                        imgo.save(outdir+newdir)
                else:
                    imgo.save(outdir+newdir)
                g=open(outdir+newbb,'a')
                g.write(str(str(cor.tolist())+','+a).replace(']','').replace('[',''))
                g.close

    elif type=="yolo":
        if l==0:
                    
                img2.save(outdir+dir.replace('.jpg','_r_'+str(name)+'.jpg'))
                g=open(outdir+bb.replace('.txt','_r_'+str(name)+'.txt'),'a')
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
                
                bobox = np.array([[xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]], np.int32)
                
                imgo,cor=rotateBox(img2,angle,bobox)
                b=cor.tolist()
                f.close()
                
                x_center = (b[0] + b[2] + b[4] + b[6]) / 4
                y_center = (b[1] + b[3] + b[5] + b[7]) / 4

                # Calculate the width and height of the bounding box
                width = max(b[0] , b[2] , b[4] , b[6]) - min(b[0] , b[2] , b[4] , b[6])
                height = max(b[1] , b[3] , b[5] ,b[7]) - min(b[1] , b[3] , b[5] ,b[7])

                # Normalize the coordinates, width, and height to be between 0 and 1
                x_center /= imgo.size[0]
                y_center /= imgo.size[1]
                width /= imgo.size[0]
                height /= imgo.size[1]
                c=[x_center,y_center,width,height]
                f=open(bb,'r')
                a=f.readlines()[i].split(' ')[0]
                f.close()
                
                imgo.save(outdir+dir.replace('.jpg','_r_'+str(name)+'.jpg'))
                g=open(outdir+bb.replace('.txt','_r_'+str(name)+'.txt'),'a')
                g.write(a+' '+str(c).replace(']','').replace('[','').replace(',',''))
                g.write('\n')
                g.close
    
def original(dir,outdir):
    img=dir
    img2=Image.open(img)
    bb=img.replace('.jpg','.txt')
    name=str(time.time())
    
    img2.save(outdir+img.replace('.jpg','_'+str(name)+'.jpg'))
    shutil.copy(bb,outdir+bb.replace('.txt','_'+str(name)+'.txt'))
    
def blur(dir,value,outdir):

    img=dir
    img2=Image.open(img)
    bb=img.replace('.jpg','.txt')
    name=str(time.time())

    img2=img2.filter(ImageFilter.GaussianBlur(value / 4))
    img2.save(outdir+img.replace('.jpg','_b_'+str(name)+'.jpg'))
    shutil.copy(bb,outdir+bb.replace('.txt','_b_'+str(name)+'.txt'))
    

def darken(dir,value,outdir):
    
    img=dir
    img2=Image.open(img)
    bb=img.replace('.jpg','.txt')
    name=str(time.time())

    img2=ImageEnhance.Brightness(img2).enhance(1.0 - value / 100)
    img2.save(outdir+img.replace('.jpg','_d_'+str(name)+'.jpg'))
    shutil.copy(bb,outdir+bb.replace('.txt','_d_'+str(name)+'.txt')) 

def brighten(dir,value,outdir):
    
    img=dir
    img2=Image.open(img)
    bb=img.replace('.jpg','.txt')
    name=str(time.time())

    img2=ImageEnhance.Brightness(img2).enhance(1.0 + value / 100)
    img2.save(outdir+img.replace('.jpg','_bi_'+str(name)+'.jpg'))
    shutil.copy(bb,outdir+bb.replace('.txt','_bi_'+str(name)+'.txt')) 
        

def flip(dir,outdir,rand):
    img=dir
    img2=Image.open(img)
    bb=img.replace('.jpg','.txt')
    name=str(time.time())

    if rand==1:
        imgo =  ImageOps.mirror(img2) #horizontal flip
    else:
        imgo = ImageOps.flip(img2) #vertical
    g=open(bb,'r')
    l=len(g.readlines())
    g.close()
    if l==0:
        imgo.save(outdir+dir.replace('.jpg','_r_'+str(name)+'.jpg'))
        g=open(outdir+bb.replace('.txt','_r_'+str(name)+'.txt'),'a')
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
            x_center /= img2.size[0]
            y_center /= img2.size[1]
            width /= img2.size[0]
            height /= img2.size[1]
            c=[x_center,y_center,width,height]
            f=open(bb,'r')
            a=f.readlines()[i].split(' ')[0]
            f.close()
        
            imgo.save(outdir+dir.replace('.jpg','_r_'+str(name)+'.jpg'))
            g=open(outdir+bb.replace('.txt','_r_'+str(name)+'.txt'),'a')
            g.write(a+' '+str(c).replace(']','').replace('[','').replace(',',''))
            g.write('\n')
            g.close
        
    
def elastic_transform(dir, value,outdir):
    
    img=dir
    img3=Image.open(img)
    img2 = cv2.cvtColor(np.array(img3), cv2.COLOR_RGB2BGR)
    bb=img.replace('.jpg','.txt')
    
    sigma=8
    
    random_state=None
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = img2.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * value
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * value
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    
    name=str(time.time())
    img2 = map_coordinates(img2, indices, order=1, mode='reflect').reshape(img2.shape)
    img4 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    img4.save(outdir+img.replace('.jpg','_e_'+str(name)+'.jpg'))
    shutil.copy(bb,outdir+bb.replace('.txt','_e_'+str(name)+'.txt'))

def rigid(dir,value,outdir):
        # ------------------------ Moving Least square ------------- rigid deformation ------------------------------------
    np.seterr(divide='ignore', invalid='ignore')

    def mls_rigid_deformation(vy, vx, p, q, alpha=1.0, eps=1e-8):

        q = np.ascontiguousarray(q.astype(np.int16))
        p = np.ascontiguousarray(p.astype(np.int16))

        # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
        p, q = q, p

        grow = vx.shape[0]  # grid rows
        gcol = vx.shape[1]  # grid cols
        ctrls = p.shape[0]  # control points

        # Compute
        reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
        reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
        
        w = 1.0 / (np.sum((reshaped_p - reshaped_v).astype(np.float32) ** 2, axis=1) + eps) ** alpha    # [ctrls, grow, gcol]
        w /= np.sum(w, axis=0, keepdims=True)                                               # [ctrls, grow, gcol]

        pstar = np.zeros((2, grow, gcol), np.float32)
        for i in range(ctrls):
            pstar += w[i] * reshaped_p[i]                                                   # [2, grow, gcol]

        vpstar = reshaped_v - pstar                                                         # [2, grow, gcol]
        reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)                                  # [2, 1, grow, gcol]
        neg_vpstar_verti = vpstar[[1, 0],...]                                               # [2, grow, gcol]
        neg_vpstar_verti[1,...] = -neg_vpstar_verti[1,...]                                  
        reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, grow, gcol)              # [2, 1, grow, gcol]
        mul_right = np.concatenate((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)    # [2, 2, grow, gcol]
        reshaped_mul_right = mul_right.reshape(2, 2, grow, gcol)                            # [2, 2, grow, gcol]

        # Calculate q
        reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
        qstar = np.zeros((2, grow, gcol), np.float32)
        for i in range(ctrls):
            qstar += w[i] * reshaped_q[i]                                                   # [2, grow, gcol]
        
        temp = np.zeros((grow, gcol, 2), np.float32)
        for i in range(ctrls):
            phat = reshaped_p[i] - pstar                                                    # [2, grow, gcol]
            reshaped_phat = phat.reshape(1, 2, grow, gcol)                                  # [1, 2, grow, gcol]
            reshaped_w = w[i].reshape(1, 1, grow, gcol)                                     # [1, 1, grow, gcol]
            neg_phat_verti = phat[[1, 0]]                                                   # [2, grow, gcol]
            neg_phat_verti[1] = -neg_phat_verti[1]
            reshaped_neg_phat_verti = neg_phat_verti.reshape(1, 2, grow, gcol)              # [1, 2, grow, gcol]
            mul_left = np.concatenate((reshaped_phat, reshaped_neg_phat_verti), axis=0)     # [2, 2, grow, gcol]
            
            A = np.matmul((reshaped_w * mul_left).transpose(2, 3, 0, 1), 
                            reshaped_mul_right.transpose(2, 3, 0, 1))                       # [grow, gcol, 2, 2]

            qhat = reshaped_q[i] - qstar                                                    # [2, grow, gcol]
            reshaped_qhat = qhat.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)            # [grow, gcol, 1, 2]

            # Get final image transfomer -- 3-D array
            temp += np.matmul(reshaped_qhat, A).reshape(grow, gcol, 2)                      # [grow, gcol, 2]

        temp = temp.transpose(2, 0, 1)                                                      # [2, grow, gcol]
        normed_temp = np.linalg.norm(temp, axis=0, keepdims=True)                           # [1, grow, gcol]
        normed_vpstar = np.linalg.norm(vpstar, axis=0, keepdims=True)                       # [1, grow, gcol]
        transformers = temp / normed_temp * normed_vpstar  + qstar                          # [2, grow, gcol]
        nan_mask = normed_temp[0] == 0

        # Replace nan values by interpolated values
        nan_mask_flat = np.flatnonzero(nan_mask)
        nan_mask_anti_flat = np.flatnonzero(~nan_mask)
        transformers[0][nan_mask] = np.interp(nan_mask_flat, nan_mask_anti_flat, transformers[0][~nan_mask])
        transformers[1][nan_mask] = np.interp(nan_mask_flat, nan_mask_anti_flat, transformers[1][~nan_mask])

        # Remove the points outside the border
        transformers[transformers < 0] = 0
        transformers[0][transformers[0] > grow - 1] = 0
        transformers[1][transformers[1] > gcol - 1] = 0
    
        return transformers.astype(np.int16)
    
        # ------------------ Return the rigid Deformation Image ---------------------------
    def demo_auto(p,q,image):  
        
        height, width,_= image.shape
        gridX = np.arange(width, dtype=np.int16)
        gridY = np.arange(height, dtype=np.int16)
        vy, vx = np.meshgrid(gridX, gridY)

        rigid = mls_rigid_deformation(vy, vx, p, q, alpha=1)
        aug3 = np.ones_like(image)
        aug3[vx, vy] = image[tuple(rigid)]
        return aug3
    # ------------------------ Function to get random coordinates with respect with given shift value -------------
    def RandMove(old_pnt,min_shift,max_shift):
        neg = [-1,1]

        #get the first point from the geometry object
        old_x = old_pnt[0]
        old_y = old_pnt[1]

        #calculate new coordinates
        new_x = old_x + (choice(neg) * randint(min_shift,max_shift))
        new_y = old_y + (choice(neg) * randint(min_shift,max_shift))

        
        return (new_x,new_y)
    
    # ------------------------------ Function to get random p and q control points -------------------------
    def check_p_q(coordinates,distance,control_points):
        p_coordinates = []
        q_coordinates = []
        while True:
            if (len(q_coordinates) == control_points):
                break
            else:
                x, y = random.choice(coordinates) 
                old_co = (x,y)
                new_co = RandMove(old_co,-distance,distance)
                # Check if the new coordinates are within the given list of coordinates
                if new_co in coordinates:
                    
                    p_coordinates.append(old_co)
                    q_coordinates.append(new_co)
                else:
                    pass
                    # print("no")
        return p_coordinates,q_coordinates
    
    #----------------------------------Function to find all the coordinates lie inside the boundary box ----------------
    def find_all_coordinates(x1,y1,x2,y2,x3,y3,x4,y4):
        # Create an empty list to hold the coordinates
        coordinates = []
        # Loop over the x values between the left and right edges of the rectangle
        for x in range(min(x1, x2, x3, x4), max(x1, x2, x3, x4) + 1):
            # Loop over the y values between the top and bottom edges of the rectangle
            for y in range(min(y1, y2, y3, y4), max(y1, y2, y3, y4) + 1):
                # Check if the current coordinate is inside the rectangle
                if (x2-x1)*(y-y1) - (y2-y1)*(x-x1) >= 0 and (x3-x2)*(y-y2) - (y3-y2)*(x-x2) >= 0 and (x4-x3)*(y-y3) - (y4-y3)*(x-x3) >= 0 and (x1-x4)*(y-y4) - (y1-y4)*(x-x4) >= 0:
                    # Append the current coordinate to the list
                    coordinates.append((x, y))               
        return coordinates
    
    def return_all_x_and_y(my_list):
        x1,y1 = int(my_list[0]),int(my_list[1])
        x2,y2 = int(my_list[2]),int(my_list[3])
        x3,y3 = int(my_list[4]),int(my_list[5])
        x4,y4 = int(my_list[6]),int(my_list[7])
        return x1,y1,x2,y2,x3,y3,x4,y4
    

    def return_from_to(path,x1,y1,x2,y2,x3,y3,x4,y4):
        im = Image.open(path)

        # Define the two points between which to find coordinates

        li = []
        # Create a new image with the same size as the original image
        new_im = Image.new('RGB', im.size, (255, 255, 255))

        # Draw a line between the two points on the new image
        draw1 = ImageDraw.Draw(new_im)
        draw1.line((x1, y1, x2, y2), fill='black')

        draw2 = ImageDraw.Draw(new_im)
        draw2.line((x2, y2, x3, y3), fill='black')

        draw3 = ImageDraw.Draw(new_im)
        draw3.line((x4, y4, x3, y3), fill='black')

        draw4 = ImageDraw.Draw(new_im)
        draw4.line((x1, y1, x4, y4), fill='black')

        # Iterate over all the pixels in the new image and print the coordinates of the black pixels
        for x in range(new_im.size[0]):
            for y in range(new_im.size[1]):
                if new_im.getpixel((x, y)) == (0, 0, 0):
                    li.append((x,y))

        select_control_points_no = int(len(li)/20)
        select_control_points = random.sample(li, select_control_points_no)
        return select_control_points

    # Open the file for reading
    path = dir
    
    img4=dir
    img3=Image.open(img4)
    img2 = cv2.cvtColor(np.array(img3), cv2.COLOR_RGB2BGR)
    bb=img4.replace('.jpg','.txt')
    
    img = img2
    my_list = []

    with open(bb, 'r') as file:
        lent = len(file.readlines())

    file.close()
    if type=='paddle':    
        for i in range(lent):
        # Open the file for reading
            with open(bb, 'r') as file:
                line = file.readlines()[i].split(',')[:-1]
                my_list.append(line)
    
    elif type=="yolo":
        f=open(bb,'r')
        x=(float(f.readlines()[0].split(" ")[1:][0]))
        f.close()

        f=open(bb,'r')
        y=(float(f.readlines()[0].split(" ")[1:][1]))
        f.close()

        f=open(bb,'r')
        w=(float(f.readlines()[0].split(" ")[1:][2]))
        f.close()

        f=open(bb,'r')
        h=(float(f.readlines()[0].split(" ")[1:][3]))
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
        
        my_list=[xmin, ymax, xmax, ymax, xmax, ymin, xmin, ymin]
    # distance= 10
    points = 10
    for i in range(len(my_list)):
        x1,y1,x2,y2,x3,y3,x4,y4 = return_all_x_and_y(my_list[i])
        all_coordinates = find_all_coordinates(x1,y1,x2,y2,x3,y3,x4,y4)
        select_control_points = return_from_to(path,x1,y1,x2,y2,x3,y3,x4,y4)
        P_Points,Q_Points = check_p_q(all_coordinates,value,points)
        #print(len(P_Points),len(Q_Points))
        # print(len(all_coordinates))
        #--------- new random points p and q
        points_in_p =  P_Points  + select_control_points
        points_in_q =  Q_Points + select_control_points
        #----- into array
        points_in_p = np.array(points_in_p)
        points_in_q = np.array(points_in_q)

        #------ points x,y into y,x-----
        for i in range(len(points_in_p)):
            # for p swap
            temp = points_in_p[i][0]
            points_in_p[i][0] = points_in_p[i][1]
            points_in_p[i][1] = temp

            #for q swap
            temp1 = points_in_q[i][0]
            points_in_q[i][0] = points_in_q[i][1]
            points_in_q[i][1] = temp1
        
        # ------------ Function called -------------
        img_deformation = demo_auto(points_in_p,points_in_q,img2)

        img_deformation = cv2.cvtColor(img_deformation,cv2.COLOR_RGB2BGR)
        img2 = img_deformation.copy()
    name=str(time.time())
    img5 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    img5.save(outdir+img4.replace('.jpg','_ri_'+str(name)+'.jpg'))
    shutil.copy(bb,outdir+bb.replace('.txt','_ri_'+str(name)+'.txt'))


        





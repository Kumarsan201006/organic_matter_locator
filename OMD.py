from imutils import paths
import cv2
import os 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import AveragePooling2D
import numpy as np
from skimage import io
from skimage.morphology import disk
from skimage.filters import median
from skimage.transform import resize
import shutil
import math
# path =input("Please enter source folder path:")

######### user input##################################
path= "D:\\Rahul\\eds\\MCS-5-10\\raw_export_2000X"
n_rows=18
######################################################

mosaic_path1= path.split("\\")

mosaic_path1.pop()
mosaic_path1.insert(1,"\\")

mosaic_path2= os.path.join(*mosaic_path1)
mosaic_path2= os.path.join(mosaic_path2, "Mosaic")

mosaic_path=mosaic_path2


if not os.path.exists(mosaic_path2):
    os.makedirs(mosaic_path2)
else:
    shutil.rmtree(mosaic_path2)           
    os.makedirs(mosaic_path2)

# n_rows= int(input("Please enter number of columns of mosaic:"))



# saving all the subfolder names in a list named "directory_contents"
directory_contents = os.listdir(path)


print("elements found")
print(directory_contents)




print("Creating Mosaic")



bse_image_list = []
for i in range(0, len(directory_contents)):
    path1=os.path.join(path, directory_contents[i]+ '\\')
    imagePaths = list(paths.list_images(path1))
    imagePaths.sort(key=len)
    
    
    image_list = []
    
    counts=[]
    # loop over the image paths, load each one, and add them to our
    # images to stitch list
    for imagePath in imagePaths:
        image = io.imread(imagePath, cv2.IMREAD_ANYDEPTH)
        image = image.astype(float)
        # image = image / image.max() #normalizes data in range 0 - 255
        # image = 255 * image
        # image = np.uint8(image)
        # img =image
        # img =img.astype(float)
        # cropping of images to remove overlapping features
        h,w= image.shape         
        crop_h= int(h*0.1)  # Overlapping is 10% on height
        crop_h= h-crop_h
        crop_w= int(w*0.1)  # Overlapping is 10% on weidth
        crop_image=image[:crop_h,crop_w:]
        
        
        
        if str(directory_contents[i])!= 'Images':
        
            # #image segemnt 0 and other pixels
            # region1 = (crop_image >= 0) & (crop_image <1)
            # region2 = (crop_image >= 1) & (crop_image <255)
            # all_regions = np.zeros((crop_image.shape[0], crop_image.shape[1])) #Create 1 channel blank image of same size as original
            # all_regions[region1] = (0)
            # all_regions[region2] = (1)
            # all_regions= all_regions.astype(np.uint8)
            # all_regions= all_regions*255
                
            
            
            h_crop,w_crop= crop_image.shape 
            
            
            # Binning of image    
            image_crop = crop_image.reshape(1, h_crop, w_crop, 1)
            binning_factor=4
            # define model containing just a single max pooling layer
            model = Sequential([AveragePooling2D(pool_size = binning_factor, strides = binning_factor)])
            # generate max-pooled output
            image_bin = model.predict(image_crop)
            # print output image
            image_bin = np.squeeze(image_bin)
            # result = ndimage.maximum_filter(crop_image, size=4, cval=0.0)   #max filter of from scipy 
            
            
            
           
            image_list.append(image_bin)
        else:
            
            bse_image_resized = resize(crop_image, (crop_image.shape[0] // 4, crop_image.shape[1] // 4),anti_aliasing=True)
            image_resized = resize(crop_image, (crop_image.shape[0] // 4, crop_image.shape[1] // 4),anti_aliasing=True)
            bse_image_list.append(bse_image_resized)
            # image_resized= image_resized.astype(np.uint8)
            filename=imagePath.split('\\')
            filename=filename[-1]
            filename=filename.split('.')
            filename=filename[0]
            filename=filename.split(' ')
            filename=filename[-1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            orgin = (50, 50) 
            fontScale = 1
            color = (0, 0, 255) 
            thickness = 3
            h1,w1= image_resized.shape 
            start_point = (1, 1)
              
            # Ending coordinate, here (220, 220)
            # represents the bottom right corner of rectangle
            
            end_point = (w1-1,h1-1)
            image_resized = cv2.putText(image_resized, filename, orgin, font,fontScale, color, thickness, cv2.LINE_AA) 
            image_resized = cv2.rectangle(image_resized, start_point, end_point, color, thickness=1)
                        
                        
            
            image_list.append(image_resized)
            
    #input the number of rows in mosaic
    # the scanning sequence was row by row start from right to left
       
    indices =[] 
    indices1 =[] 
    for a in range (0,(len(image_list)+n_rows),n_rows):
        indices1.append(a)
    
    
    b=0
    c=b+1
    vertical_list=[]
    try:
        while c<(len(image_list)+1):
            for d in range(indices1[b],indices1[c]):
                # print("d=",d)
                indices.append(d)
                selected_elements = []
                for index in indices:
                    selected_elements.append(image_list[index])
                    
                    
                selected_elements.reverse()
                my_array = np.array(selected_elements)
                my_array1=np.hstack((my_array))
                            
                
            vertical_list.append(my_array1)   
            indices.clear()
            selected_elements.clear()   
                # plt.imshow(my_array1)
                
            
            b=b+1
            c=c+1
            # print("b=",b)
            # print("c=",c)
        
    except:
        pass
        # print("all images stacked")
    # vertical_list1=np.array(vertical_list)
    
    
    # vertical_list2=np.resize(vertical_list1, (42160,37020))
    
    
    final1=np.vstack((vertical_list))
    # values_mosaic, counts_mosaic = np.unique(final1, return_counts=True)
    # print(values_mosaic)
    # print(counts_mosaic)
    
    
    
    if str(directory_contents[i])!= 'Images':
        
    
        image_median = median(final1, disk(3))
        
        image_median = image_median / image_median.max() #normalizes data in range 0 - 255
        image_median = 255 * image_median
        image_median = np.uint8(image_median)
        
        
        
        
               
            
        region1 = (image_median ==0)
        region2 = (image_median !=0)
       
        all_regions = np.zeros((image_median.shape[0], image_median.shape[1],3)) #Create 3 channel blank image of same size as original
        all_regions[region1] = (0,0,0)
        all_regions[region2] = (255,0,0)
        
        # plt.imshow(final)
        print("Saving "+directory_contents[i]+" mosaic images")
        output_path= os.path.join(mosaic_path, directory_contents[i]+'.png')
        io.imsave(output_path,all_regions )
    
            
            # image_median = Image.fromarray(np.uint8(cm.gist_earth(final1)*255))
        # rgba = image_median.convert('RGB')

        # datas = rgba.getdata()
        
        
        # newData = []
        
        # for item in datas:
        
        #    if item[0] == 0 and item[1] == 0 and item[2] == 0:
        
        #        newData.append((255, 255, 255, 0))
        
        #    else:
        
        #       newData.append(item)
              
              
        # rgba.putdata(newData)

        # rgba.save(output_path, 'PNG')
        
    
    else:
        print("Saving"+directory_contents[i]+" mosaic images")
        
        final1 = final1 / final1.max() #normalizes data in range 0 - 255
        final1 = 255 * final1
        final1 = np.uint8(final1)
        
        output_path= os.path.join(mosaic_path, directory_contents[i]+'.png')
        
        
        io.imsave(output_path,final1 )



indices =[] 
indices1 =[] 
for a in range (0,(len(bse_image_list)+n_rows),n_rows):
    indices1.append(a)
    
b=0
c=b+1
vertical_list=[]
try:
    while c<(len(bse_image_list)+1):
        for d in range(indices1[b],indices1[c]):
            # print("d=",d)
            indices.append(d)
            selected_elements = []
            for index in indices:
                selected_elements.append(bse_image_list[index])
                
                
            selected_elements.reverse()
            my_array = np.array(selected_elements)
            my_array1=np.hstack((my_array))
                        
            
        vertical_list.append(my_array1)   
        indices.clear()
        selected_elements.clear()   
            # plt.imshow(my_array1)
            
        
        b=b+1
        c=c+1
        # print("b=",b)
        # print("c=",c)
    
except:
    print("all images stacked")
final1=np.vstack((vertical_list))
final1 = final1 / final1.max() #normalizes data in range 0 - 255
final1 = 255 * final1
final1 = np.uint8(final1)
# final1=final1.astype('uint8')
output_path1= os.path.join(mosaic_path,'BSE.png')
io.imsave(output_path1,final1 )


c_image_path= os.path.join(mosaic_path,'C.png')
c_image= cv2.imread(c_image_path,0)
values_c, counts_c = np.unique(c_image, return_counts=True) 

if values_c[1]!=0:
    c_image = c_image / c_image.max() #normalizes data in range 0 - 255
    
    
bse_image_path= os.path.join(mosaic_path,'BSE.png')
bse_image= cv2.imread(bse_image_path)


ca_image_path= os.path.join(mosaic_path,'Ca.png')
ca_image= cv2.imread(ca_image_path,0)


values_ca, counts_ca = np.unique(ca_image, return_counts=True) 
if values_ca[1]!=0:
    ca_image = ca_image / ca_image.max()


iom= c_image-ca_image
iom= iom+1
iom= iom/iom.max()
iom= np.uint8(iom)
iom= iom*255
region = (iom >= 200) & (iom <=255)


all_regions = np.zeros((iom.shape[0], iom.shape[1], 3)) #Create 3 channel blank image of same size as original

all_regions[region] = (1,0,0)
# all_regions= np.uint8(all_regions)
iom_image_path= os.path.join(mosaic_path,'IOM.png')
io.imsave(iom_image_path,all_regions)
overlay= cv2.imread(iom_image_path)

dest = cv2.addWeighted(overlay, 1, bse_image, 1, 0.0)

# Start coordinate, here (5, 5)
# represents the top left corner of rectangle
# start point
height= int(dest.shape[0]/10)
width= int(dest.shape[1]/4)

# start_point = (0,0)
start_point = (dest.shape[1]-width, dest.shape[0]-height)
  
# Ending coordinate, here (220, 220)
# represents the bottom right corner of rectangle
end_point = (dest.shape[1],dest.shape[0])
  
# # Blue color in BGR
# color = (0, 0, 0)
  
# # Line thickness of 2 px
thickness = -1
  
# # Using cv2.rectangle() method
# # Draw a rectangle with blue line borders of thickness of 2 px
# dest = cv2.rectangle(dest, start_point, end_point, color, thickness)
 


# for scale bar
# start_point = (0,0)
start_point_sbar = (start_point[0]+int(start_point[0]*0.03),start_point[1]+int(start_point[1]*0.03))
  
# Ending coordinate, here (220, 220)
# represents the bottom right corner of rectangle
end_point_sbar = (end_point[0]-int(end_point[0]*0.03),end_point[1]-int(end_point[1]*0.05))

# Blue color in BGR
color_sbar = (0,0,0)
dest = cv2.rectangle(dest, start_point_sbar, end_point_sbar, color_sbar, thickness)

#for scale bar

len_sbar= end_point_sbar[0]-start_point_sbar[0] # len in pixels   2048pixesl=244 um
len_sbar= str(int(len_sbar*(61*binning_factor/2048)))  # len in um

# org
org = (start_point_sbar[0], start_point_sbar[1]-(end_point_sbar[1]-start_point_sbar[1]))
  
# text
text= len_sbar+"um"

def optimal_font_dims(img, font_scale = 2e-3, thickness_scale = 5e-3):
    h, w, _ = dest.shape
    font_scale = min(w, h) * font_scale
    thickness = math.ceil(min(w, h) * thickness_scale)
    return font_scale, thickness

font_scale, thickness = optimal_font_dims(dest)
cv2.putText(dest, text, org, cv2.FONT_HERSHEY_SIMPLEX, int(font_scale/2), (0,0,255), thickness)
io.imshow(dest) 

filaname=  os.path.join(mosaic_path,'IOM_overlay_BSE.png')

cv2.imwrite(filaname, dest)

# values, counts = np.unique(ca_image, return_counts=True) 
# print(values,counts)

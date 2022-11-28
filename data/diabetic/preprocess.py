import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob

def main(mode): 
    
    paths = glob(f'./{mode}/*')
        
    for img_ph in paths:
        save_path = f"./{mode}_process/{os.path.basename(img_ph)}"
        if os.path.exists(save_path):
            continue
                            
        if not os.path.exists(img_ph):
            with open(f'./{mode}_exclude.txt', 'a') as file:
                file.write(f"{img_ph}\n")
            continue
                      
        raw_image = cv2.imread(img_ph)
        
        if raw_image is None:
            with open(f'./{mode}_exclude.txt', 'a') as file:
                file.write(f"{img_ph}\n")
            continue

#         raw_image = cv2.fastNlMeansDenoisingColored(raw_image, None, 1, 1, 7, 21)
    
        size_wide = raw_image.shape[1]
        
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        c_x, c_y, c_r, thres_t = get_circle_mask(raw_image)
        
        if c_x==0:
            with open(f'./{mode}_exclude.txt', 'a') as file:
                file.write(f"{img_ph}\n")
            continue

        im_cropped, ratio, mask = get_cropped_Im(raw_image, c_x, c_y, c_r)
        im_crop_nor = get_normalized_Im(im_cropped, 51, mask)  

        cv2.imwrite(save_path, im_crop_nor)
            
def get_circle_mask(im):

    c_x, c_y, c_r = 0,0,0

    th, thIm = get_mask(im)

    x, y = np.where(thIm == 255)  

    if np.prod(x.shape) > 0:
        c_x = np.mean(x, axis=0, dtype=np.int)
        c_y = np.mean(y, axis=0, dtype=np.int)

    edge_im = cv2.Canny(thIm,0,255)   

    e_x, e_y = np.where(edge_im > 0)

    if np.prod(e_x.shape) > 0:   
        r = np.sort(np.square(e_x - c_x)+np.square(e_y - c_y) )
        c_r = np.sqrt(r[int(np.prod(r.shape)*0.5)])-1
         
    else:
        c_r = np.min([c_x, c_y])-1

    if c_r > np.max([im.shape[0], im.shape[1]]) / 2:
        c_r = np.min([c_x, c_y])-1

    return c_x, c_y, np.int_(c_r), th


def get_mask(im):

    avg_im = np.uint8((np.float32(im[:,:,0]) + np.float32(im[:,:,1]) + np.float32(im[:,:,2]))/3)

    hist, bins = np.histogram(avg_im.flatten(),256,[0,256])

    cdf = hist.cumsum()

    threshold_max = 20

    proba = np.true_divide(hist, cdf)

    best_threshold, min_delta = 0, 100000

    for i in range(1,threshold_max):
        if cdf[i] > 0:
            delta = float(hist[i+1]) / float(cdf[i])
            if delta < min_delta:
                min_delta = delta
                best_threshold = i

    ret, mask = cv2.threshold(avg_im,best_threshold,255,cv2.THRESH_BINARY)

    return best_threshold, mask

def get_cropped_Im(im, c_x, c_y, c_r):  

    crop_imSize = int( 2*c_r + 1)
    tem_im = np.zeros((crop_imSize, crop_imSize, 3))

    x0, x1, y0, y1 = c_r, c_r, c_r, c_r    
    if c_x - c_r < 0: 
        x0 = c_x
    if c_y - c_r < 0: 
        y0 = c_y
    if c_x + c_r > im.shape[0]:
        x1 = im.shape[0] - c_x
    if c_y + c_r > im.shape[1]:
        y1 = im.shape[1] - c_y    
  
    tem_im[c_r-x0:c_r+x1,c_r-y0:c_r+y1,:] = im[c_x-x0:c_x+x1,c_y-y0:c_y+y1,:]    

    x_arr = np.zeros((1,crop_imSize))
    y_arr = np.transpose(x_arr)
    x_arr[0,:] = np.arange(crop_imSize)    
    dis_xy = np.square(x_arr - c_r + 1)+np.square(y_arr - c_r + 1)
    tem_im[dis_xy>=c_r*c_r,:] = 0

    resized_image = np.uint8(cv2.resize(tem_im, (512, 512)))#(1024, 1024)
    
    th, resized_mask = get_mask(resized_image)

    ratio = crop_imSize / 512.0

    return resized_image, ratio, resized_mask

def get_normalized_Im(im, kernal_size, mask):
 

    im_r = np.int_(im[:,:, 0])
    im_g = np.int_(im[:,:, 1])
    im_b = np.int_(im[:,:, 2])
    mask = np.int_(mask)

    mask_add = np.zeros((im_r.shape[0],im_r.shape[1]))
    mask_add[im_r >= 220] = mask_add[im_r >= 220] + 1
    mask_add[im_g <= 30] = mask_add[im_g <= 30] + 1
    mask_add[im_b >= 220] = mask_add[im_b >= 220] + 1

    mask[mask > 0] = 1

    mask[mask_add == 3] = 0

    kernel = np.ones((kernal_size,kernal_size), np.float64) #/(kernal_size*kernal_size)

    mean_r = cv2.filter2D(im_r,-1,kernel)
    mean_g = cv2.filter2D(im_g,-1,kernel)
    mean_b = cv2.filter2D(im_b,-1,kernel)
    cnt_mask = cv2.filter2D(mask,-1,kernel)

    mean_r[mask==0] = 0
    mean_g[mask==0] = 0
    mean_b[mask==0] = 0

    cnt_mask[mask == 0] = 1

    mean_r = np.true_divide(mean_r, cnt_mask)
    mean_g = np.true_divide(mean_g, cnt_mask)
    mean_b = np.true_divide(mean_b, cnt_mask)

    dis_r = im_r - mean_r
    dis_g = im_g - mean_g
    dis_b = im_b - mean_b

    # --- compute mean and stddev for each channel
    stat_r = dis_r[mask == 1]
    stat_g = dis_g[mask == 1]
    stat_b = dis_b[mask == 1]

    stat_r_mean = np.mean(stat_r)
    stat_g_mean = np.mean(stat_g)
    stat_b_mean = np.mean(stat_b)

    stat_r_std = np.std(stat_r)
    stat_g_std = np.std(stat_g)
    stat_b_std = np.std(stat_b)


    im_r = (dis_r - stat_r_mean)/stat_r_std
    im_r[im_r>3.0] = 3.0
    im_r[im_r<-3.0] = -3.0
    im_r = (im_r + 3.0)/6.0*255.0+0.5
    im_r[im_r >255] = 255

    im_g = (dis_g - stat_g_mean)/stat_g_std
    im_g[im_g>3.0] = 3.0
    im_g[im_g<-3.0] = -3.0
    im_g = (im_g + 3.0)/6.0*255.0+0.5
    im_g[im_g >255] = 255

    im_b = (dis_b - stat_b_mean)/stat_b_std
    im_b[im_b>3.0] = 3.0
    im_b[im_b<-3.0] = -3.0
    im_b = (im_b + 3.0)/6.0*255.0+0.5
    im_b[im_b >255] = 255

    im[:,:,0] = im_r
    im[:,:,1] = im_g
    im[:,:,2] = im_b

    im[mask == 0,0] = 0
    im[mask == 0,1] = 0
    im[mask == 0,2] = 0
    im = cv2.medianBlur(np.uint8(im),3)

    return im 

if __name__ == '__main__':
    main('train')
    main('test')

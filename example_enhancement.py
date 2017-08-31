# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:19:20 2017

@author: Shawn Yuen
"""
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

# parameters
r = 16
eps = 0.01

# boxfilter
def boxfilter(imSrc, r):
    (hei, wid) = imSrc.shape
    imDst = np.zeros_like(imSrc)
    
    imCum = np.cumsum(imSrc, 0)
    imDst[0:r+1, :] = imCum[r:2*r+1, :]
    imDst[r+1:hei-r, :] = imCum[2*r+1:hei, :] - imCum[0:hei-2*r-1, :]
    imDst[hei-r:hei, :] = np.tile(imCum[hei-1, :], [r,1]) - imCum[hei-2*r-1:hei-r-1,:]
    
    imCum = np.cumsum(imDst, 1)
    imDst[:, 0:r+1] = imCum[:, r:2*r+1]
    imDst[:, r+1:wid-r] = imCum[:, 2*r+1:wid] - imCum[:, 0:wid-2*r-1]
    imDst[:, wid-r:wid] = np.tile(imCum[:, wid-1], [r,1]).T - imCum[:, wid-2*r-1:wid-r-1]
    #imDst[:, cols-r:cols] = np.tile(imCum[:, cols-1], [1, r]) - imCum[:, cols-2*r-1:cols-r-1]
    
    return imDst

# for grayscale
def guidedfilter(I, p, r, eps):
    (hei, wid) = I.shape
    N = boxfilter(np.ones([hei, wid]), r)
    
    mean_I = boxfilter(I, r) / N
    mean_p = boxfilter(p, r) / N
    mean_Ip = boxfilter(I*p, r) / N
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = boxfilter(I*I, r) / N
    var_I = mean_II - mean_I*mean_I
    
    a = cov_Ip / (var_I + eps) #http://kaiminghe.com/publications/eccv10guidedfilter.pdf, eqn. (5)
    b = mean_p - a * mean_I #http://kaiminghe.com/publications/eccv10guidedfilter.pdf, eqn. (6)
    
    mean_a = boxfilter(a, r) / N
    mean_b = boxfilter(b, r) / N
    
    q = mean_a * I + mean_b #http://kaiminghe.com/publications/eccv10guidedfilter.pdf, eqn. (8)
    return q
    
# for color
def guidedfilter_color(I, p, r, eps):
    (hei, wid) = p.shape
    N = boxfilter(np.ones([hei, wid]), r)
    
    mean_I_r = boxfilter(I[:,:,0], r) / N
    mean_I_g = boxfilter(I[:,:,1], r) / N
    mean_I_b = boxfilter(I[:,:,2], r) / N
    
    mean_p = boxfilter(p, r) / N
    
    mean_Ip_r = boxfilter(I[:,:,0]*p, r) / N
    mean_Ip_g = boxfilter(I[:,:,1]*p, r) / N
    mean_Ip_b = boxfilter(I[:,:,2]*p, r) / N
    
    cov_Ip_r = mean_Ip_r - mean_I_r * mean_p
    cov_Ip_g = mean_Ip_g - mean_I_g * mean_p
    cov_Ip_b = mean_Ip_b - mean_I_b * mean_p
    
    var_I_rr = boxfilter(I[:,:,0]*I[:,:,0], r) / N - mean_I_r * mean_I_r
    var_I_rg = boxfilter(I[:,:,0]*I[:,:,1], r) / N - mean_I_r * mean_I_g
    var_I_rb = boxfilter(I[:,:,0]*I[:,:,2], r) / N - mean_I_r * mean_I_b
    var_I_gg = boxfilter(I[:,:,1]*I[:,:,1], r) / N - mean_I_g * mean_I_g
    var_I_gb = boxfilter(I[:,:,1]*I[:,:,2], r) / N - mean_I_g * mean_I_b
    var_I_bb = boxfilter(I[:,:,2]*I[:,:,2], r) / N - mean_I_b * mean_I_b
    
    a = np.zeros([hei, wid, 3])
    for y in range(hei):
        for x in range(wid):
            Sigma = [var_I_rr[y,x], var_I_rg[y,x], var_I_rb[y,x], var_I_rg[y,x], var_I_gg[y,x], var_I_gb[y,x], var_I_rb[y,x], var_I_gb[y,x], var_I_bb[y,x]]
            Sigma = np.reshape(Sigma, [3, 3])
            #print(Sigma)
            cov_Ip = [cov_Ip_r[y,x], cov_Ip_g[y,x], cov_Ip_b[y,x]]
            
            a[y, x, :] = cov_Ip * np.mat(Sigma + eps * np.eye(3)).I #http://kaiminghe.com/publications/eccv10guidedfilter.pdf, eqn. (14)
            #a[y, x, :] = cov_Ip * np.mat(Sigma).I
    
    b = mean_p - a[:,:,0] * mean_I_r - a[:,:,1] * mean_I_g - a[:,:,2] * mean_I_b
    q = boxfilter(a[:,:,0], r)*I[:,:,0] + boxfilter(a[:,:,1], r)*I[:,:,1] + boxfilter(a[:,:,2], r)*I[:,:,2] + boxfilter(b, r)
    q = q / N
    return q
    
def main():
    img_name = './img_enhancement/tulips.bmp'
    img_enhanced_name = './img_enhancement/tulips_enhanced.bmp'
    print("Loading image", img_name)
    img = scipy.misc.imread(img_name).astype(np.float64)/255
    (hei, wid, dim) = img.shape
    q = np.zeros([hei, wid, dim])
    # check img and mask
    if True:
        plt.figure(1)
        plt.imshow(img)
        #print(img.shape,img.dtype)
    p = img
    q[:,:,0] = guidedfilter(img[:,:,0], p[:,:,0], r, eps)
    q[:,:,1] = guidedfilter(img[:,:,1], p[:,:,1], r, eps)
    q[:,:,2] = guidedfilter(img[:,:,2], p[:,:,2], r, eps)
    img_enhanced = (img-q)*5 + q
    scipy.misc.imsave(img_enhanced_name, img_enhanced)
    plt.figure(2)
    plt.imshow(q, vmin=0, vmax=1)
    plt.figure(3)
    plt.imshow(img_enhanced, vmin=0, vmax=1)
    
if __name__ == "__main__":
    main()
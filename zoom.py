import cv2
import numpy as np


def replication(img, f):
    r = len(img)
    c = len(img[0])
    res = np.zeros(shape=(f*r, f*c))
    ri = ci = 0
    for i in img:
        ci = 0
        for j in i:
            res[ri][ci] = res[ri+1][ci] = res[ri][ci+1] = res[ri+1][ci+1] = j
            ci += 2
        ri += 2
    return res


def zoh(img, f):
    r = len(img)
    c = len(img[0])
    res = np.zeros(shape=(f*r-1, f*c-1))
    ri = ci = 0
    for i in range(0, r):
        ci = 0
        for j in range(0, c):
            res[ri][ci] = img[i][j]
            if j+1 < c and ci+1 < f*c-1:
                res[ri][ci+1] = int((int(img[i][j])+int(img[i][j+1]))/2)
            ci += 2
        ri += 2
    ri = ci = 0
    for i in range(0, f*r-1):
        ci = 0
        for j in range(0, f*c-1):
            if i+2 < f*r-1 and ri+1 < f*r-1:
                res[ri+1][ci] = int((int(res[i][j])+int(res[i+2][j]))/2)
            ci += 1
        ri += 2
    return res


def ktz(img, k):
    r = len(img)
    c = len(img[0])
    res = np.zeros(shape=(k*(r-1)+1, k*(c-1)+1))
    ri = ci = 0
    for i in range(0, r):
        ci = 0
        for j in range(0, c):
            res[ri][ci] = img[i][j]
            t = k-1
            v = img[i][j]-img[i][j+1]
            if v < 0:
                v *= -1
            op = v
            while(t):
                ci += 1
                op += int(v/k)
                res[ri][ci] = op
                t -= 1
        ri += t

        ri += 2


img = cv2.imread('img10.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
r = np.array(r)
b = np.array(b)
g = np.array(g)
f = 2
r = replication(r, f)
g = replication(g, f)
b = replication(b, f)
r = cv2.UMat(np.array(r, dtype=np.uint8))
g = cv2.UMat(np.array(g, dtype=np.uint8))
b = cv2.UMat(np.array(b, dtype=np.uint8))
image = cv2.merge((b, g, r))
#res2 = zoh(img, f)
# print(img1)
# print(res2)
#res2 = cv2.UMat(np.array(res2, dtype=np.uint8))
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

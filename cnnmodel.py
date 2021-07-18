import cv2
import numpy as np
import matplotlib.pyplot as plt


def average(img):
    avg = 0
    n = 0
    for i in img:
        for j in i:
            if j != 0:
                avg += int(j)
                n += 1
                #print(str(j)+" "+str(n))
    return avg//n


image = cv2.imread("img10.jpg")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
print(pixel_values.shape)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

k = 2
ret, labels, (centers) = cv2.kmeans(pixel_values, k, None,
                                    criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
labels = labels.flatten()
print(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)
cluster = 1
masked_image = np.copy(image)
masked_image = masked_image.reshape((-1, 3))
masked_image[labels == cluster] = [0, 0, 0]
masked_image = masked_image.reshape(image.shape)
masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
print(average(gray))
avg = average(gray)
binary = cv2.threshold(gray, avg, 255, cv2.THRESH_BINARY_INV)[1]
lab = cv2.cvtColor(masked_image, cv2.COLOR_BGR2LAB)
a = lab[:, :, 1]
avg1 = average(a)
# plt.imshow(a)
# plt.show()
# plt.imshow(lab)
# plt.show()
bw_img = cv2.threshold(a, avg1, 255, cv2.THRESH_BINARY)[1]
out_img1 = cv2.bitwise_and(masked_image, masked_image, mask=binary)
out_img2 = cv2.bitwise_and(masked_image, masked_image, mask=bw_img)
#cv2.imshow('image', out_img1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
plt.imshow(out_img1)
plt.show()

#cv2.imshow('image', out_img2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
plt.imshow(out_img2)
plt.show()

img = np.asarray(out_img2)
l, b, h = img.shape

scale = 1
aa = np.zeros((scale*(l-1)+1, scale*(b-1)+1, h))
aa1 = np.zeros((scale*(l-1)+1, scale*(b-1)+1, h))

j1 = 0

for i in range(0, l):
    for j in range(0, b-1):
        for k in range(0, h):
            if (img[i][j][k] <= img[i][j+1][k]):
                op = img[i][j+1][k] - img[i][j][k]
                op = op/scale
                for n in range(0, scale):
                    aa[i][j1+n][k] = img[i][j][k]+(op*n)
            else:
                op = img[i][j][k] - img[i][j+1][k]
                op = op/scale
                for n in range(0, scale):
                    aa[i][j1+n][k] = img[i][j][k]-(op*n)
        j1 = j1+scale
    aa[i][j1][:] = img[i][j+1][:]
    j1 = 0

i1 = 0

for j in range(0, scale*(b-1)+1):
    for i in range(0, l-1):
        for k in range(0, h):
            if (aa[i][j][k] <= aa[i+1][j][k]):
                op = aa[i+1][j][k] - aa[i][j][k]
                op = op/scale
                for n in range(0, scale):
                    aa1[i1+n][j][k] = aa[i][j][k]+(op*n)
            else:
                op = aa[i][j][k] - aa[i+1][j][k]
                op = op/scale
                for n in range(0, scale):
                    aa1[i1+n][j][k] = aa[i][j][k]-(op*n)
        i1 = i1+scale
    aa1[i1][j][:] = aa[i+1][j][:]
    i1 = 0
x = 190
y = 108
x1 = int((x*scale) - l/2)
x2 = int((x*scale) + l/2)
y1 = int((y*scale) - b/2)
y2 = int((y*scale) + b/2)

if (x1 < 0):
    x2 = x2-x1
    x1 = 0
if (x2 > (l*scale)):
    x1 = x1 - (x2-(l*scale))
    x2 = l*scale


if (y1 < 0):
    y2 = y2-y1
    y1 = 0

if (y2 > (b*scale)):
    y1 = y1 - (y2-(b*scale))
    y2 = b*scale


zoomed_image = aa1[x1:x2, y1:y2]
result = cv2.UMat(np.array(zoomed_image, dtype=np.uint8))
cv2.imshow('image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

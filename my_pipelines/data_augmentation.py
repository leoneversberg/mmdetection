from mmdet.datasets import PIPELINES
import imgaug.augmenters as iaa #conda install imgaug
import cv2
import numpy as np

@PIPELINES.register_module()
class MyDataAugmentation(object):
    #def __init__(self):
    #    self.asd = 1

    #@staticmethod
    def augmentation(self,img):
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
        # image.

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        seq = iaa.Sequential([
            sometimes(iaa.JpegCompression(compression=(1, 30))),
            sometimes(iaa.GaussianBlur(sigma=(0.2, 1.0))),
            sometimes(iaa.MultiplyHue((0.9, 1.1))),
            sometimes(iaa.AdditiveGaussianNoise(scale=0.01*255, per_channel=0.5))
        ], random_order=True)
        img_aug = seq(image=img)

        #cv2.imwrite("asd.jpg", img_aug)
        return img_aug

    def __call__(self, results):
        results['img'] = self.augmentation(results['img'])
        #results['dummy'] = True
        return results



@PIPELINES.register_module()
class PencilFilter(object):
    #def __init__(self):
    #    self.asd = 1

    #@staticmethod
    #def filter(self,img):
    #    G  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #    kernel = np.ones((5,5),np.uint8)
    #    P = cv2.dilate(G,kernel,iterations = 1)
    #    m,n = G.shape
    #    for i in range(m):
    #        for j in range(n):
    #            if (P[i,j] == 0):
    #                P[i,j] = 255
    #            else:
    #                P[i,j] = np.uint8(G[i,j] * 255/P[i,j])
    #    P = cv2.cvtColor(P, cv2.COLOR_GRAY2BGR)
    #    cv2.imwrite("asd.jpg", P)
    #    return P
    def create_pencil_image(self,input_image, dilate_size = 2):

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_size + 1, 2 * dilate_size + 1),
                                            (dilate_size, dilate_size))

        pencil_image = cv2.dilate(input_image, element)

        valorig = input_image.copy()
        valmax = pencil_image.copy()
        valout = pencil_image.copy()
        alpha_image = pencil_image.copy()

        valout[valmax == 0] = 255

        alpha_image[valmax > 0] = (valorig[valmax>0]*float(255.0)) / valmax[valmax > 0]

        valout[valout < 255] = alpha_image[valout < 255]

        pencil_image = valout.copy()
        pencil_image = np.expand_dims(pencil_image, axis=2)
        pencil_image = cv2.cvtColor(pencil_image, cv2.COLOR_GRAY2BGR)
        #cv2.imwrite("asd.jpg", pencil_image)


        return pencil_image

    def __call__(self, results):
        results['img'] = self.create_pencil_image(results['img'])
        #results['dummy'] = True
        return results

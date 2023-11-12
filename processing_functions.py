from skimage.exposure import rescale_intensity
from skimage.transform import rescale, rotate
import cv2

def rescale_crop_image(img):
  # rescale short side to standard size, then crop center
  standard = 256
  scale = standard / min(img.shape[:2])
  img = rescale(img, scale, anti_aliasing=True, channel_axis = 2)
  img = img[int(img.shape[0]/2 - standard/2) : int(img.shape[0]/2 + standard/2),
            int(img.shape[1]/2 - standard/2) : int(img.shape[1]/2 + standard/2)]
  return img

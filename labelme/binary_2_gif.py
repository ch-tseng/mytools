import cv2, imageio
import glob, os

path = r'/DS/working/Pytorch-UNet/data/src/3-2.aug/aug_masks/*.jpg'
output = r'/DS/working/Pytorch-UNet/data/src/4.gif/'

path = path.replace('\\', '/')
output = output.replace('\\', '/')

if not os.path.exists(output):
	os.makedirs(output)

for imgfile in glob.glob(path):
	print(imgfile)
	image_lst = []
    
	im_gray = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
	(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	image_lst.append(im_bw)
	fname = os.path.basename(imgfile)
	filename = os.path.splitext(fname)[0] + '.gif'
	#cv2.imwrite( os.path.join(output, filename), im_bw)
	imageio.mimsave(os.path.join(output, filename), image_lst, fps=60)

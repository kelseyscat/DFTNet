import os
import gdal
import matplotlib.pyplot as plt

def show_image(imgpath, band=999):
    img = gdal.Open(imgpath)
    bands = img.RasterCount
    img_width = img.RasterXSize
    img_height = img.RasterYSize
    imgmata = img.GetMetadataItem("PHOTOMETRIC", "DMD_CREATIONOPTIONLIST")
    print("bands:", bands)
    print("img_width:", img_width)
    print("img_height:", img_height)
    img_data = img.ReadAsArray()
    if len(img_data.shape) < 3:
        print("this image just has one band ")
        print("img_data:", img_data)
        plt.figure('landsat: img_data')
        plt.imshow(img_data)
        plt.show()
        return
    if band == 999:
        print("please enter a band number! example:show_image_band(img_x,3).")
        return
    if band >= bands:
        print("out range of bands, it should be < ", bands)
        return
    print("show image in band " + str(band))
    img1_band = img_data[band, 0:img_height, 0:img_width]
    plt.figure('landsat: img_peer_band')
    plt.imshow(img1_band)
    plt.show()

if __name__ == '__main__':
    root_path = "CIA/train"
    for folder in os.listdir(root_path):
        p = root_path + "/" + folder
        for i in os.listdir(p):
            path = p + "/" + i
            show_image(path, 0)
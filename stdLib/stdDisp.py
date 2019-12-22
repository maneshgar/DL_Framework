import matplotlib.pyplot as plt

def gridShow(images, sizeX, sizeY, rows, cols, titles=None, cmap=plt.cm.binary):
    plt.figure(figsize=(sizeX, sizeY))
    for i in range(rows * cols):
        plt.subplot(rows,cols,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=cmap)
        if titles is not None:
            plt.xlabel(titles[i])
    plt.show()

###################################################################################
###################################################################################

def imageShow(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()


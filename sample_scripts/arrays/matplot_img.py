
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

# First, load the image
#dir_path = os.path.dirname(os.path.realpath(__file__))
filename = "../../images/MarshOrchid.jpg"
image = mpimg.imread(filename)

print(filename)
# Print out its shape
print(image.shape)

#plt.imshow(image)
#plt.show()

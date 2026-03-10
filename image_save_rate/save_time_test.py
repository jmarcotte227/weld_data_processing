from PIL import Image
import timeit
import numpy as np

image = np.ones((512,640), dtype=np.float32)
def save_random_img():
    for i in range(100):
        im = Image.fromarray(image)
        im.save(f"image_{i}.tiff")

if __name__=="__main__":
    print(timeit.timeit("save_random_img()",number=1, globals=globals())/100)




import os
path = "C:\\Users\\alessio\\PycharmProjects\\cable_segmentation\\file\\input\\large\\normal"

for r, d, images in os.walk(path):
    for img in images:
        p = os.path.join(r, img)
        p.replace('normal', 'label')
        if not os.path.exists(p):
            print("ERRORE {}".format(p))

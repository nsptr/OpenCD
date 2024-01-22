from PIL import Image

infile = '(B060)_2022_37709093.tif'
outfile = infile[:-3]+'jpeg'
im = Image.open(infile)

out = im.convert("RGB")
out.save(outfile, "jpeg", quality=90)


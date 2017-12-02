## Fall 2017: GPUs Architecture and Programming

### Final Project: Total Variation Denoising 

The function takes a bitmap image (*.ppm) as an input argument. The function takes the image, adds Gaussian white noise and performs total variation denoising. The output are one noisy image and one denoised image. There is an image included in the file ```C bicycle.ppm```. If you are interested in running the code with this image, simply type:

```C
nvcc -o tv TV_denoise.cu ppma_io.c -lm
./tv bicycle.ppm
```

If you are interested in running the code with some other images, make sure to convert the image into *.ppm by typing:
```C
convert image.jpg -compress none image.ppm
```

[Report](http://example.com)




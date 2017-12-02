## Fall 2017: GPUs Architecture and Programming

### Final Project: Total Variation Denoising 

The function takes a bitmap image (\texttt{\*.ppm}) as an input argument. The function takes the image, adds Gaussian white noise and performs total variation denoising. The output are one noisy image and one denoised image. There is an image included in the file \texttt{bicycle.ppm}. If you are interested in running the code with this image, simply type: \\
\texttt{nvcc -o tv TV\_denoise.cu ppma\_io.c -lm} \\
\texttt{./tv bicycle.ppm} \\
If you are interested in running the code with some other images, make sure to convert the image into \texttt{\*.ppm} by typing:\\
\texttt{convert image.jpg -compress none image.ppm}

[Report](http://example.com)




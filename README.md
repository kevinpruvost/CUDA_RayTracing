# CUDA_RayTracing
Personal Project on implementing Ray Tracing with CUDA, visualizable on an OpenGL Renderer

## Resources

[CUDA/OpenGL Interoperability Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html)
[CUDA Architecture Code Matching for different setups](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

## Notes

- When working with multiple files that need external symbol links, don't forget (on Visual Studio) to activate Properties/Cuda C/C++/Common/Generate Relocatable Device Code.

- I am working with an RTX 3070 Laptop, so the GPU is using Ampere architecture, sm_86 should be used then.

- For the number of threads and blocks, I will just go with 16x16 threads, and a number of blocks based on the number of pixels to render divided by the number of threads, so `(width + block_size - 1) / block_size` and `(height + block_size - 1) / block_size`.
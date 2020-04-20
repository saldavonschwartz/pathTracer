//
//  PathTracer.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "PathTracer.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include "Utils.hpp"
#include "Ray.hpp"
#include "Scene.hpp"
#include "Material.hpp"
#include "Camera.hpp"
#include "HittableVector.hpp"
#include "BVH.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using std::ofstream;
using std::string;
using std::cerr;


#define USE_GPU 

gvec3 sampleRay(const Ray& ray, float tmin, float tmax, int maxBounces, Hittable& hittable) {
	if (maxBounces <= -1) {
		return {};
	}

	Hittable::HitInfo info;

	if (hittable.hit(ray, tmin, tmax, info)) {
		gvec3 attenuation;
		Ray scattered;

		if (info.material->scatter(ray, info, attenuation, scattered)) {
			return attenuation * sampleRay(scattered, tmin, tmax, maxBounces - 1, hittable);
		}

		return {};
	}
	
	float t = 0.5f * (ray.dir + gvec3{ 1.f, 1.f, 1.f }).y;
	return (1.f - t) * gvec3 {1.f, 1.f, 1.f } +t * gvec3{ .5f, .7f, 1.f };
}

__host__ __device__ gvec3 sampleRayIterative(const Ray& ray, float tmin, float tmax, int maxBounces, Hittable& hittable) {
	gvec3 color;

	while (maxBounces >= -1) {
		maxBounces -= 1;
		Hittable::HitInfo info;

		if (hittable.hit(ray, tmin, tmax, info)) {
			gvec3 attenuation;
			Ray scattered;

			if (info.material->scatter(ray, info, attenuation, scattered)) {
				color *= attenuation;
			}
			else {
				color = { 0.f, 0.f, 0.f };
				break;
			}
		}
		else {
			float t = 0.5f * (ray.dir + gvec3{ 1.f, 1.f, 1.f }).y;
			color = (1.f - t) * gvec3 { 1.f, 1.f, 1.f } +t * gvec3{ .5f, .7f, 1.f };
			break;
		}
	}

	return color;
}

#ifndef USE_GPU

int renderScene(int sceneId, string path, int width, int height, int raysPerPixel, int maxBouncesPerRay) {
  ofstream outputImage(path + "out2.ppm");
  
  if (!outputImage.is_open()) {
    cerr << "ERROR: could not open output file!\n";
    return -1;
  }
  
  float aspectRatio = double(width) / height;
  vec3 lookFrom;
  vec3 lookAt;
  float focalLength;
  float aperture;
  float fovy;
  HittableVector sceneObjects;
  
  if (sceneId == 1) {
    lookFrom = {3.f, 3.f, 2.f};
    lookAt = {0.f, 0.f, -1.f};
    focalLength = length(lookFrom-lookAt);
    aperture = 0.1;
    fovy = 20;
    sceneObjects = generateSimpleScene();
  }
  else {
    lookFrom = {13.f, 2.f, 3.f};
    lookAt = vec3{0.f};
    focalLength = 10.f;
    aperture = 0.1f;
    fovy = 20.f;
    sceneObjects = generateComplexScene();
  }
  
  HittableVector world;
  Camera cam(lookFrom, lookAt, fovy, aspectRatio, focalLength, aperture);
  BVHNode scene(sceneObjects, 0.f, 1.f);
  int rowsProcessed  = 0;
  
  // Output image is in PPM 'plain' format (http://netpbm.sourceforge.net/doc/ppm.html#plainppm)
  outputImage << "P3\n" << width << " " << height << "\n255\n";
  
  // Image origin is bottom-left.
  // Pixels are output one row at a time, top to bottom, left to right:
  {
    auto p = Profiler("[Render Time]");
    
    for (int y = height-1; y >= 0; y--) {
      if (!rowsProcessed) {
        cerr << std::fixed << std::setprecision(1) <<
        "\n" << 100.f*(1.f - (float)y/height) << "% complete" <<
        std::flush;
      }
      
      rowsProcessed = (rowsProcessed + 1) % 20;
      
      for (int x = 0; x < width; x++) {
        vec3 pixel{0};
        
        for (int r = 0; r < raysPerPixel; r++) {
          float u = (x + urand(0, 1))/float(width);
          float v = (y + urand(0, 1))/float(height);
          Ray ray = cam.castRay(u, v);
          pixel += sampleRay(ray, 0.001f, inf, maxBouncesPerRay, scene);
        }
        
        pixel = 255.f * glm::clamp(glm::sqrt(pixel/float(raysPerPixel)), 0.f, 1.f);
        outputImage << int(pixel.r) << " " << int(pixel.g) << " " << int(pixel.b) << "\n";
      }
    }
  }
  
  cerr << "\n100% complete" << std::flush;
  outputImage.close();
  return 0;
}
#else

#define CHK_CUDA(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(-1);
	}
}

__global__
void gpuRender(gvec3* buffer, int width, int height) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if ((x >= width) || (y >= height)) return;
	int pixel_index = y * width + x;
	buffer[pixel_index] = gvec3(float(x) / width, float(y) / height, 0.2);
}

int renderScene(int sceneId, string path, int width, int height, int raysPerPixel, int maxBouncesPerRay) {
	cerr << "\n[Using CUDA]\n";

	ofstream outputImage(path + "imgOutCuda.ppm");

	if (!outputImage.is_open()) {
		cerr << "ERROR: could not open output file!\n";
		return -1;
	}

	/*float aspectRatio = double(width) / height;
	vec3 lookFrom;
	vec3 lookAt;
	float focalLength;
	float aperture;
	float fovy;
	HittableVector sceneObjects;

	if (sceneId == 1) {
		lookFrom = { 3.f, 3.f, 2.f };
		lookAt = { 0.f, 0.f, -1.f };
		focalLength = length(lookFrom - lookAt);
		aperture = 0.1;
		fovy = 20;
		sceneObjects = generateSimpleScene();
	}
	else {
		lookFrom = { 13.f, 2.f, 3.f };
		lookAt = vec3{ 0.f };
		focalLength = 10.f;
		aperture = 0.1f;
		fovy = 20.f;
		sceneObjects = generateComplexScene();
	}

	HittableVector world;
	Camera cam(lookFrom, lookAt, fovy, aspectRatio, focalLength, aperture);
	BVHNode scene(sceneObjects, 0.f, 1.f);
	*/
	size_t fBufferSize = sizeof(float) * 3 * width * height;
	gvec3* fBuffer = nullptr;
	CHK_CUDA(cudaMallocManaged(&fBuffer, fBufferSize));
	dim3 threads(8, 8);
	dim3 blocks(width / threads.x + 1, height / threads.y + 1);

	{
		auto p = Profiler("[Render Time]");

		gpuRender << <blocks, threads >> > (fBuffer, width, height);
		CHK_CUDA(cudaGetLastError());
		CHK_CUDA(cudaDeviceSynchronize());
	}

	// Output image is in PPM 'plain' format (http://netpbm.sourceforge.net/doc/ppm.html#plainppm)
	outputImage << "P3\n" << width << " " << height << "\n255\n";

	// Image origin is bottom-left.
	// Pixels are output one row at a time, top to bottom, left to right:
	
	for (int y = height - 1; y >= 0; y--) {
		for (int x = 0; x < width; x++) {
			/*for (int r = 0; r < raysPerPixel; r++) {
				float u = (x + urand(0, 1)) / float(width);
				float v = (y + urand(0, 1)) / float(height);
				Ray ray = cam.castRay(u, v);
				pixel += sampleRay(ray, 0.001f, inf, maxBouncesPerRay, scene);
			}*/

			int p = y * width + x;
			gvec3 pixel = 255.99f * fBuffer[p];
			
			//pixel = 255.f * glm::clamp(glm::sqrt(pixel / float(raysPerPixel)), 0.f, 1.f);
			outputImage << int(pixel.r) << " " << int(pixel.g) << " " << int(pixel.b) << "\n";
		}
	}

	CHK_CUDA(cudaFree(fBuffer));
	cerr << "\n100% complete" << std::flush;
	outputImage.close();
	return 0;
}

#endif


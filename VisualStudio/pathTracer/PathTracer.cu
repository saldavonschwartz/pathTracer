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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "Utils.hpp"
#include "Ray.hpp"
#include "Scene.hpp"
#include "Material.hpp"
#include "Camera.hpp"
#include "HittableVector.hpp"
#include "BVH.hpp"

using std::ofstream;
using std::string;
using std::cerr;

__device__ gvec3 sampleRay(Ray ray, float tmin, float tmax, int maxBounces, const Hittable& hittable,  curandState* rState) {
	gvec3 color{1.f};
	gvec3 attenuation;

	while (maxBounces > -1) {
		Hittable::HitInfo info;

		if (hittable.hit(ray, tmin, tmax, info)) {		
			if (info.material->scatter(ray, info, attenuation, ray, rState)) {
				color *= attenuation;
				maxBounces -= 1;
			}
			else {
				color *= {};
				break;
			}
		}
		else {
			float t = 0.5f * (ray.dir + gvec3{ 1.f, 1.f, 1.f }).y;
			color *= (1.f - t) * gvec3 { 1.f, 1.f, 1.f } +t * gvec3{ .5f, .7f, 1.f };
			break;
		}
	}

	return color;
}

__global__ void randInit(int w, int h, curandState* randState) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= w || y >= h) {
		return;
	}

	int p = y * w + x;
	curand_init(1984, p, 0, &randState[p]);
}

__global__ void freeScene(HittableVector* scene, Camera* cam) {
	delete scene;
	delete cam;
}

__global__
void renderScene(
	gvec3* frameBuff, int w, int h, Camera* cam, Hittable* scene, 
	int raysPerPixel, int maxBouncesPerRay, curandState* rStates
) 
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= w || y >= h) {
		return;
	}

	int p = y * w + x;
	curandState rState = rStates[p];
	gvec3 pixel;

	for (int r = 0; r < raysPerPixel; r++) {
		float u = (x + curand_uniform(&rState)) / float(w);
		float v = (y + curand_uniform(&rState)) / float(h);
		Ray ray = cam->castRay(u, v, &rState);
		pixel += sampleRay(ray, 0.001f, 10000.f, maxBouncesPerRay, *scene, &rState);
	}

	pixel /= float(raysPerPixel);
	pixel.r = sqrt(pixel.r);
	pixel.g = sqrt(pixel.g);
	pixel.b = sqrt(pixel.b);
	frameBuff[p] = 255.f * pixel;
}

int renderScene(int sceneId, string path, int width, int height, int raysPerPixel, int maxBouncesPerRay) {
	cerr << "\n[Using CUDA]\n";

	ofstream outputImage(path + "imgOutCuda.ppm");

	if (!outputImage.is_open()) {
		cerr << "ERROR: could not open output file!\n";
		return -1;
	}

	float aspect = double(width) / height;
	auto sceneCam = generateSimpleScene(aspect);
	HittableVector* scene = sceneCam.first;
	Camera* cam = sceneCam.second;
	//BVHNode scene(sceneObjects, 0.f, 1.f);
	
	size_t pixelCount = width * height;
	dim3 threads(8, 8);
	dim3 blocks(width / threads.x + 1, height / threads.y + 1);

	// Frame Buffer:
	size_t fbSize = sizeof(float) * 3 * pixelCount;
	gvec3* frameBuff = nullptr;
	CHK_CUDA(cudaMallocManaged(&frameBuff, fbSize));
	
	// Random:
	curandState *perPixelRand;
	CHK_CUDA(cudaMalloc(&perPixelRand, sizeof(curandState) * pixelCount));
	
	randInit << <blocks, threads >> > (width, height, perPixelRand);
	CHK_CUDA(cudaGetLastError());
	CHK_CUDA(cudaDeviceSynchronize());

	// Render:
	{
		auto p = Profiler("[Render Time]");
		
		renderScene << <blocks, threads >> > (frameBuff, width, height, cam, scene, raysPerPixel, maxBouncesPerRay, perPixelRand);
		CHK_CUDA(cudaGetLastError());
		CHK_CUDA(cudaDeviceSynchronize());
	}

	// Save File (CPU):
	// Output image is in PPM 'plain' format (http://netpbm.sourceforge.net/doc/ppm.html#plainppm)
	outputImage << "P3\n" << width << " " << height << "\n255\n";

	// Image origin is bottom-left.
	// Pixels are output one row at a time, top to bottom, left to right:
	
	for (int y = height - 1; y >= 0; y--) {
		for (int x = 0; x < width; x++) {
			int p = y * width + x;
			gvec3 pixel = frameBuff[p];
			outputImage 
				<< int(pixel.r) << " " 
				<< int(pixel.g) << " " 
				<< int(pixel.b) << "\n";
		}
	}

	CHK_CUDA(cudaFree(frameBuff));
	cerr << "\n100% complete" << std::flush;
	outputImage.close();
	return 0;
}




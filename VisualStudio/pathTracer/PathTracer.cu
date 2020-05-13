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
#include "Material.hpp"
#include "Camera.hpp"
#include "HittableVector.hpp"
#include "BVH.hpp"
#include "Scene.hpp"
#include "Hittable.hpp"

using std::ofstream;
using std::string;
using std::cerr;

__device__ gvec3 sampleRay(Ray ray, float tmin, float tmax, int maxBounces, Hittable* hittable, curandState* rState) {
	gvec3 color{ 1.f };
	gvec3 attenuation;

	while (maxBounces > -1) {
		Hittable::HitInfo info;
		
		if (hittable->hit(ray, tmin, tmax, info)) {
			if (info.material->scatter(ray, info, attenuation, ray, rState)) {
				color = info.material->emission() + color * attenuation;
				maxBounces -= 1;
			}
			else {
				color *= info.material->emission();
				break;
			}
		}
		else {
			color *= {0.f};
			float t = 0.5f * (ray.dir + gvec3{ 1.f, 1.f, 1.f }).y;
			color *= (1.f - t) * gvec3 { 1.f, 1.f, 1.f } +t * gvec3{ .5f, .7f, 1.f };
			break;
		}
	}

	return color;
}

__global__ void randInit(int w, int h, curandState* rStates) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= w || y >= h) {
		return;
	}

	int p = y * w + x;
	curand_init(1984 + p, 0, 0, &rStates[p]);
}

__global__ void renderScene(
	gvec3* frameBuff, int w, int h, Camera* cam, Hittable** scene,
	int raysPerPixel, int maxBouncesPerRay, curandState* rStates
)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= w || y >= h) {
		return;
	}

	int p = y * w + x;
	curandState* rs = &rStates[p];
	gvec3 pixel;

	for (int r = 0; r < raysPerPixel; r++) {
		float u = (x + curand_uniform(rs)) / float(w);
		float v = (y + curand_uniform(rs)) / float(h);
		Ray ray = cam->castRay(u, v, rs);
		pixel += sampleRay(ray, 0.001f, inf, maxBouncesPerRay, *scene, rs);
	}

	pixel /= float(raysPerPixel);
	pixel.r = sqrt(pixel.r);
	pixel.g = sqrt(pixel.g);
	pixel.b = sqrt(pixel.b);
	frameBuff[p] = 255.f * pixel;
}

int renderScene(string path, int width, int height, int raysPerPixel, int maxBouncesPerRay) {
	float aspect = float(width) / height;
	size_t pixelCount = width * height;
	dim3 threads(8, 8);
	dim3 blocks(width / threads.x + 1, height / threads.y + 1);
	//cudaError_t e = cudaDeviceSetLimit(cudaLimitStackSize, 800 * sizeof(BVHNode));

	// Generate scene + camera:
	BVHNode** scene;
	Camera* cam;
	generateScene(scene, cam, aspect);

	// w * h randoms to render (one per pixel):
	curandState *perPixelRand;
	CHK_CUDA(cudaMalloc(&perPixelRand, sizeof(curandState) * pixelCount));
	
	randInit << <blocks, threads >> > (width, height, perPixelRand);
	CHK_CUDA(cudaGetLastError());
	CHK_CUDA(cudaDeviceSynchronize());

	// alloc frame buffer:
	gvec3* fBuffer;
	CHK_CUDA(cudaMallocManaged(&fBuffer, sizeof(float) * 3 * pixelCount));

	// Render:
	renderScene << <blocks, threads >> > (
		fBuffer, width, height, cam, (Hittable**)scene, 
		raysPerPixel, maxBouncesPerRay, perPixelRand
	);

	CHK_CUDA(cudaGetLastError());
		
	{
		auto p = Profiler("[Render Time]");
		CHK_CUDA(cudaDeviceSynchronize());
	}

	freeScene(scene, cam);
	
	ofstream outputImage(path);

	if (!outputImage.is_open()) {
		cerr << "ERROR: could not open output file!\n";
		return -1;
	}

	// Save file as ascii PPM:
	// http://netpbm.sourceforge.net/doc/ppm.html#plainppm
	outputImage << "P3\n" << width << " " << height << "\n255\n";

	// Image origin is bottom-left.
	// Pixels are output one row at a time, top to bottom, left to right:

	for (int y = height - 1; y >= 0; y--) {
		for (int x = 0; x < width; x++) {
			int p = y * width + x;
			gvec3 pixel = fBuffer[p];
			outputImage
				<< int(pixel.r) << " "
				<< int(pixel.g) << " "
				<< int(pixel.b) << "\n";
		}
	}

	outputImage.close();

	CHK_CUDA(cudaFree(perPixelRand));
	CHK_CUDA(cudaFree(fBuffer));

	return 0;
}

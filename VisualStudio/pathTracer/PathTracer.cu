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
#include "Sphere.hpp"

using std::ofstream;
using std::string;
using std::cerr;

__device__ gvec3 sampleRay(Ray ray, float tmin, float tmax, int maxBounces, Hittable* hittable,  curandState* rState) {
	gvec3 color{1.f};
	gvec3 attenuation;

	while (maxBounces > -1) {
		Hittable::HitInfo info;

		if (hittable->hit(ray, tmin, tmax, info)) {		
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

__global__ void generateSimpleScene(HittableVector** scene, Camera* cam, float aspect) {
	*scene = new HittableVector();
	(*scene)->init(5);
	(*scene)->add(new Sphere(gvec3(0.f, 0.f, -1.f), 0.5f, new Diffuse(gvec3(0.1f, 0.2f, 0.5f))));
	(*scene)->add(new Sphere(gvec3(0.f, -100.5f, -1.f), 100.f, new Diffuse(gvec3(0.8f, 0.8f, 0.0f))));
	(*scene)->add(new Sphere(gvec3(1.f, 0.f, -1.f), 0.5f, new Metal(gvec3(0.8f, 0.6f, 0.2f), 0.3f)));
	(*scene)->add(new Sphere(gvec3(-1.f, 0.f, -1.f), 0.5f, new Dielectric(1.5f)));
	(*scene)->add(new Sphere(gvec3(-1.f, 0.f, -1.f), -0.45f, new Dielectric(1.5f)));

	gvec3 lookFrom = { 3.f, 3.f, 2.f };
	gvec3 lookAt = { 0.f, 0.f, -1.f };
	float f = length(lookFrom - lookAt);
	float a = 0.1f;
	float fovy = 20.f;
	cam->init(lookFrom, lookAt, fovy, aspect, f, a);
}

__global__ void generateSimpleScene2(HittableVector* scene, Camera* cam, float aspect) {
	scene->init(5);
	scene->add(new Sphere(gvec3(0.f, 0.f, -1.f), 0.5f, new Diffuse(gvec3(0.1f, 0.2f, 0.5f))));
	scene->add(new Sphere(gvec3(0.f, -100.5f, -1.f), 100.f, new Diffuse(gvec3(0.8f, 0.8f, 0.0f))));
	scene->add(new Sphere(gvec3(1.f, 0.f, -1.f), 0.5f, new Metal(gvec3(0.8f, 0.6f, 0.2f), 0.3f)));
	scene->add(new Sphere(gvec3(-1.f, 0.f, -1.f), 0.5f, new Dielectric(1.5f)));
	scene->add(new Sphere(gvec3(-1.f, 0.f, -1.f), -0.45f, new Dielectric(1.5f)));

	gvec3 lookFrom = { 3.f, 3.f, 2.f };
	gvec3 lookAt = { 0.f, 0.f, -1.f };
	float f = length(lookFrom - lookAt);
	float a = 0.1f;
	float fovy = 20.f;
	cam->init(lookFrom, lookAt, fovy, aspect, f, a);
}

__global__ void generateComplexScene(
	BVHNode** bvh, HittableVector** scene, Camera* cam, float aspect, curandState* rState) {
	if (threadIdx.x != 0 || blockIdx.x != 0) {
		return;
	}

	curandState rs = *rState;
	int x = 10, y = 10;
	int size = ((x * 2) * (y * 2)) + 4;
	
	*scene = new HittableVector();
	(*scene)->init(size);
	
	(*scene)->add(new Sphere(gvec3{ 0.f,-1000.f,0.f }, 1000.f, new Diffuse(gvec3{ 0.5f })));
	(*scene)->add(new Sphere(gvec3{ 0.f, 1.f, 0.f }, 1.f, new Dielectric(1.5f)));
	(*scene)->add(new Sphere(gvec3{ -4.f, 1.f, 0.f }, 1.f, new Diffuse(gvec3{ 0.4f, 0.2f, 0.1f })));
	(*scene)->add(new Sphere(gvec3{ 4.f, 1.f, 0.f }, 1.f, new Metal(gvec3{ 0.7f, 0.6f, 0.5f }, 0.f)));

	for (int a = -x; a < x; a++) {
		for (int b = -y; b < y; b++) {
			auto materialProbability = curand_uniform(&rs);
			gvec3 center{ a + 0.9f * curand_uniform(&rs), 0.2f, b + 0.9f * curand_uniform(&rs) };

			if (length(center - gvec3{ 4.f, 0.2f, 0.f }) > 0.9f) {
				if (materialProbability < 0.8f) {
					// Diffuse
					gvec3 albedo = urand3(&rs) * urand3(&rs);
					(*scene)->add(new Sphere(center, 0.2f, new Diffuse(albedo)));
				}
				else if (materialProbability < 0.95f) {
					// Metal
					gvec3 albedo = (urand3(&rs) + 1.f) * 0.5f;
					auto fuzziness = curand_uniform(&rs) * 0.5f;
					(*scene)->add(new Sphere(center, 0.2f, new Metal(albedo, fuzziness)));
				}
				else {
					// glass
					(*scene)->add(new Sphere(center, 0.2f, new Dielectric(1.5f)));
				}
			}
		}
	}
	
	//*bvh = new BVHNode(*scene, 0.f, 1.f, &rs);
	*rState = rs;

	gvec3 lookFrom = { 13.f, 2.f, 3.f };
	gvec3 lookAt = { 0.f };
	float f = 10.f;
	float a = 0.1f;
	float fovy = 20.f;

	cam->init(lookFrom, lookAt, fovy, aspect, f, a);
}

__global__ void randInit(int w, int h, curandState* randState) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= w || y >= h) {
		return;
	}

	int p = y * w + x;
	curand_init(1984+p, 0, 0, &randState[p]);
}

__global__ void randInit1(curandState *rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1984, 0, 0, rand_state);
	}
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
	curandState rs = rStates[p];
	gvec3 pixel;

	for (int r = 0; r < raysPerPixel; r++) {
		float u = (x + curand_uniform(&rs)) / float(w);
		float v = (y + curand_uniform(&rs)) / float(h);
		Ray ray = cam->castRay(u, v, &rs);
		pixel += sampleRay(ray, 0.001f, inf, maxBouncesPerRay, *scene, &rs);
	}

	rStates[p] = rs;

	pixel /= float(raysPerPixel);
	pixel.r = sqrt(pixel.r);
	pixel.g = sqrt(pixel.g);
	pixel.b = sqrt(pixel.b);
	frameBuff[p] = 255.f * pixel;
}

__global__ void freeScene(Hittable** scene) {
	delete (*scene);
}

int renderScene(int sceneId, string path, int width, int height, int raysPerPixel, int maxBouncesPerRay) {
	ofstream outputImage(path + "imgOutCuda.ppm");

	if (!outputImage.is_open()) {
		cerr << "ERROR: could not open output file!\n";
		return -1;
	}

	float aspect = float(width) / height;
	size_t pixelCount = width * height;
	dim3 threads(8, 8);
	dim3 blocks(width / threads.x + 1, height / threads.y + 1);

	cudaDeviceReset();

	// One random to generate the scene:
	curandState *sceneGenRand;
	CHK_CUDA(cudaMalloc(&sceneGenRand, sizeof(curandState)));
	randInit1 << <1, 1 >> > (sceneGenRand);
	CHK_CUDA(cudaGetLastError());
	CHK_CUDA(cudaDeviceSynchronize());

	// w * h randoms to render (one per pixel):
	curandState *perPixelRand;
	CHK_CUDA(cudaMalloc(&perPixelRand, sizeof(curandState) * pixelCount));
	randInit << <blocks, threads >> > (width, height, perPixelRand);
	CHK_CUDA(cudaGetLastError());
	CHK_CUDA(cudaDeviceSynchronize());

	// Generate scene + camera:
	HittableVector** scene;
	CHK_CUDA(cudaMalloc(&scene, sizeof(HittableVector*)));

	BVHNode** bvh;
	CHK_CUDA(cudaMalloc(&bvh, sizeof(BVHNode*)));

	Camera* cam;
	CHK_CUDA(cudaMalloc(&cam, sizeof(Camera)));

	 //generateSimpleScene2 << <1, 1 >> > (scene, cam, aspect);
	generateComplexScene << <1, 1 >> > (bvh, scene, cam, aspect, sceneGenRand);
	CHK_CUDA(cudaGetLastError());
	CHK_CUDA(cudaDeviceSynchronize());
	//BVHNode scene(sceneObjects, 0.f, 1.f);
	
	// alloc frame buffer:
	gvec3* fBuffer;
	CHK_CUDA(cudaMallocManaged(&fBuffer, sizeof(float) * 3 * pixelCount));
	
	// Render:
	renderScene << <blocks, threads >> > (fBuffer, width, height, cam, (Hittable**)scene, raysPerPixel, maxBouncesPerRay, perPixelRand);	
	CHK_CUDA(cudaGetLastError());
		
	{
		auto p = Profiler("[Render Time]");
		CHK_CUDA(cudaDeviceSynchronize());
	}

	{
		auto p = Profiler("[Save Time]");

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
	}

	freeScene << <1, 1 >> > ((Hittable**)scene);
	CHK_CUDA(cudaGetLastError());
	CHK_CUDA(cudaDeviceSynchronize());

	CHK_CUDA(cudaFree(fBuffer));
	CHK_CUDA(cudaFree(scene));
	CHK_CUDA(cudaFree(cam));
	return 0;
}




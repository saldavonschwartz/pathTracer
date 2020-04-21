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

#define CHK_CUDA(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(-1);
	}
}

__host__ __device__ gvec3 sampleRay(Ray ray, float tmin, float tmax, int maxBounces, const Hittable& hittable) {
	gvec3 color{1.f};
	gvec3 attenuation;

	while (maxBounces > -1) {
		Hittable::HitInfo info;

		if (hittable.hit(ray, tmin, tmax, info)) {		
			if (info.material->scatter(ray, info, attenuation, ray)) {
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

__global__
void gpuRender(gvec3* fb, int w, int h, const Camera& cam, int maxBounces) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= w || y >= h) {
		return;
	}

	float u = float(x) / w;
	float v = float(y) / h;
	int p = y * w + x;
	fb[p] = sampleRay(cam.castRay(u, v), 0.f, inf, maxBounces, hittable);
}

int renderScene(int sceneId, string path, int width, int height, int raysPerPixel, int maxBouncesPerRay) {
	cerr << "\n[Using CUDA]\n";

	ofstream outputImage(path + "imgOutCuda.ppm");

	if (!outputImage.is_open()) {
		cerr << "ERROR: could not open output file!\n";
		return -1;
	}

	float aspectRatio = double(width) / height;
	gvec3 lookFrom;
	gvec3 lookAt;
	float focalLength;
	float aperture;
	float fovy;
	HittableVector sceneObjects;

	if (sceneId == 1) {
		lookFrom = { 3.f, 3.f, 2.f };
		lookAt = { 0.f, 0.f, -1.f };
		focalLength = length(lookFrom - lookAt);
		aperture = 0.1f;
		fovy = 20.f;
		sceneObjects = generateSimpleScene();
	}
	else {
		lookFrom = { 13.f, 2.f, 3.f };
		lookAt = {};
		focalLength = 10.f;
		aperture = 0.1f;
		fovy = 20.f;
		sceneObjects = generateComplexScene();
	}

	HittableVector world;
	Camera cam(lookFrom, lookAt, fovy, aspectRatio, focalLength, aperture);
	BVHNode scene(sceneObjects, 0.f, 1.f);
	
	size_t fbSize = sizeof(float) * 3 * width * height;
	gvec3* fb = nullptr;
	CHK_CUDA(cudaMallocManaged(&fb, fbSize));
	dim3 threads(8, 8);
	dim3 blocks(width / threads.x + 1, height / threads.y + 1);

	{
		auto p = Profiler("[Render Time]");

		gpuRender << <blocks, threads >> > (fb, width, height, cam);
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



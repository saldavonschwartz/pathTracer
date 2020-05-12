//
//  Scene.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "Scene.hpp"
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "HittableVector.hpp"
#include "Camera.hpp"
#include "Sphere.hpp"
#include "Material.hpp"
#include "BVH.hpp"

//__global__ void genScene1(BVHNode** bvh, HittableVector** scene, Camera* cam, float aspect, curandState* rState) {
//	curand_init(1984, 0, 0, rState);
//	curandState rs = *rState;
//	
//	*scene = new HittableVector();
//	(*scene)->init(5);
//	
//	(*scene)->add(new Sphere(gvec3(0.f, 0.f, -1.f), 0.5f, new Diffuse(gvec3(0.1f, 0.2f, 0.5f))));
//	(*scene)->add(new Sphere(gvec3(0.f, -100.5f, -1.f), 100.f, new Diffuse(gvec3(0.8f, 0.8f, 0.0f))));
//	(*scene)->add(new Sphere(gvec3(1.f, 0.f, -1.f), 0.5f, new Metal(gvec3(0.8f, 0.6f, 0.2f), 0.3f)));
//	(*scene)->add(new Sphere(gvec3(-1.f, 0.f, -1.f), 0.5f, new Dielectric(1.5f)));
//	(*scene)->add(new Sphere(gvec3(-1.f, 0.f, -1.f), -0.45f, new Dielectric(1.5f)));
//	
//	//*bvh = new BVHNode(*scene, 0.f, 1.f, &rs);
//	*rState = rs;
//
//	gvec3 lookFrom = { 3.f, 3.f, 2.f };
//	gvec3 lookAt = { 0.f, 0.f, -1.f };
//	float f = length(lookFrom - lookAt);
//	float a = 0.1f;
//	float fovy = 20.f;
//	cam->init(lookFrom, lookAt, fovy, aspect, f, a);
//}

__global__ void _generateScene(BVHNode** bvh, Camera* cam, float aspect, curandState* rs) {
	int x = 10, y = 10;
	int capacity = ((x * 2) * (y * 2)) + 4;
	HittableVector scene(capacity);

	scene.add(new Sphere(gvec3{ 0.f,-1000.f,0.f }, 1000.f, new Diffuse(gvec3{ 0.5f })));
	scene.add(new Sphere(gvec3{ 0.f, 1.f, 0.f }, 1.f, new Dielectric(1.5f)));
	scene.add(new Sphere(gvec3{ -4.f, 1.f, 0.f }, 1.f, new Diffuse(gvec3{ 0.4f, 0.2f, 0.1f })));
	scene.add(new Sphere(gvec3{ 4.f, 1.f, 0.f }, 1.f, new Metal(gvec3{ 0.7f, 0.6f, 0.5f }, 0.f)));

	for (int a = -x; a < x; a++) {
		for (int b = -y; b < y; b++) {
			auto materialProbability = curand_uniform(rs);
			gvec3 center{ a + 0.9f * curand_uniform(rs), 0.2f, b + 0.9f * curand_uniform(rs) };

			if (length(center - gvec3{ 4.f, 0.2f, 0.f }) > 0.9f) {
				if (materialProbability < 0.8f) {
					// Diffuse
					gvec3 albedo = urand3(rs) * urand3(rs);
					scene.add(new Sphere(center, 0.2f, new Diffuse(albedo)));
				}
				else if (materialProbability < 0.95f) {
					// Metal
					gvec3 albedo = (urand3(rs) + 1.f) * 0.5f;
					auto fuzziness = curand_uniform(rs) * 0.5f;
					scene.add(new Sphere(center, 0.2f, new Metal(albedo, fuzziness)));
				}
				else {
					// glass
					scene.add(new Sphere(center, 0.2f, new Dielectric(1.5f)));
				}
			}
		}
	}

	*bvh = new BVHNode(&scene, 0.f, 1.f, rs);
	
	gvec3 lookFrom = { 13.f, 2.f, 3.f };
	gvec3 lookAt = { 0.f };
	float f = 10.f;
	float a = 0.1f;
	float fovy = 20.f;

	cam->init(lookFrom, lookAt, fovy, aspect, f, a);
}

__global__ void initRand(curandState* rState) {
	curand_init(1984, 0, 0, rState);
}

__global__ void _freeScene(BVHNode** bvh) {
	delete *bvh;
}

void generateScene(BVHNode**& bvh, Camera*& cam, float aspect) {
	CHK_CUDA(cudaMalloc(&bvh, sizeof(BVHNode*)));
	CHK_CUDA(cudaMalloc(&cam, sizeof(Camera)));

	curandState* rState;
	CHK_CUDA(cudaMalloc(&rState, sizeof(curandState)));
	
	initRand << <1, 1 >> > (rState);
	CHK_CUDA(cudaGetLastError());
	CHK_CUDA(cudaDeviceSynchronize());

	_generateScene << <1, 1 >> > (bvh, cam, aspect, rState);
	CHK_CUDA(cudaGetLastError());
	CHK_CUDA(cudaDeviceSynchronize());
	CHK_CUDA(cudaFree(rState));
}

void freeScene(BVHNode**& bvh, Camera*& cam) {
	_freeScene << <1, 1 >> > (bvh);
	CHK_CUDA(cudaGetLastError());
	CHK_CUDA(cudaDeviceSynchronize());
	CHK_CUDA(cudaFree(bvh));
	CHK_CUDA(cudaFree(cam));
}


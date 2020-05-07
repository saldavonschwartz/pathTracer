//
//  Scene.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "Scene.hpp"
#include "Sphere.hpp"
#include "Material.hpp"
#include "Utils.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

using std::make_shared;

__global__ void genSimpleScene(HittableVector* scene, Camera** cam, float aspect) {
	scene->data = new Hittable*[5];
	scene->push_back(new Sphere(gvec3(0, 0, -1), 0.5, new Diffuse(gvec3(0.1, 0.2, 0.5))));
	scene->push_back(new Sphere(gvec3(0, -100.5, -1), 100, new Diffuse(gvec3(0.8, 0.8, 0.0))));
	scene->push_back(new Sphere(gvec3(1, 0, -1), 0.5, new Metal(gvec3(0.8, 0.6, 0.2), 0.3)));
	scene->push_back(new Sphere(gvec3(-1, 0, -1), 0.5, new Dielectric(1.5)));
	scene->push_back(new Sphere(gvec3(-1, 0, -1), -0.45, new Dielectric(1.5)));

	gvec3 lookFrom = { 3.f, 3.f, 2.f };
	gvec3 lookAt = { 0.f, 0.f, -1.f };
	float f = length(lookFrom - lookAt);
	float a = 0.1f;
	float fovy = 20.f;
	*cam = new Camera(lookFrom, lookAt, fovy, aspect, f, a);
}

std::pair<HittableVector*, Camera*> generateSimpleScene(float aspect) {
	HittableVector* scene;
	CHK_CUDA(cudaMalloc((void**)&scene, sizeof(HittableVector*)));

	Camera* cam;
	CHK_CUDA(cudaMalloc((void**)&cam, sizeof(Camera*)));

	genSimpleScene << <1, 1 >> > (scene, &cam, aspect);
	CHK_CUDA(cudaGetLastError());
	CHK_CUDA(cudaDeviceSynchronize());
	return {scene, cam};
}

//__global__ void genComplexScene(HittableVector& scene, curandState& randState) {
//	HittableVector scene;
//	scene.push_back(new Sphere(gvec3{ 0.f,-1000.f,0.f }, 1000.f, new Diffuse(gvec3{ 0.5f })));
//
//	for (int a = -10; a < 10; a++) {
//		for (int b = -10; b < 10; b++) {
//			auto materialProbability = curand_uniform(&randState);
//			gvec3 center{ a + 0.9f * curand_uniform(&randState), 0.2f, b + 0.9f * curand_uniform(&randState) };
//
//			if (length(center - gvec3{ 4.f, 0.2f, 0.f }) > 0.9f) {
//				if (materialProbability < 0.8f) {
//					// Diffuse
//					auto albedo = glm::sphericalRand(1.) * glm::sphericalRand(1.);
//					scene.push_back(new Sphere(center, 0.2f, new Diffuse(albedo)));
//				}
//				else if (materialProbability < 0.95f) {
//					// Metal
//					auto albedo = glm::sphericalRand(.5f) + 0.5f;
//					auto fuzziness = curand_uniform(&randState);
//					scene.push_back(new Sphere(center, 0.2f, new Metal(albedo, fuzziness)));
//				}
//				else {
//					// glass
//					scene.push_back(new Sphere(center, 0.2f, new Dielectric(1.5f)));
//				}
//			}
//		}
//	}
//
//	scene.push_back(new Sphere(gvec3{ 0.f, 1.f, 0.f }, 1.f, new Dielectric(1.5f)));
//	scene.push_back(new Sphere(gvec3{ -4.f, 1.f, 0.f }, 1.f, new Diffuse(gvec3{ 0.4f, 0.2f, 0.1f })));
//	scene.push_back(new Sphere(gvec3{ 4.f, 1.f, 0.f }, 1.f, new Metal(gvec3{ 0.7f, 0.6f, 0.5f }, 0.f)));
//}
//
//HittableVector generateComplexScene() {
//	Hittable** data;
//	CHK_CUDA(cudaMalloc((void**)&data, 400 * sizeof(Hittable*)));
//
//	HittableVector scene(data, 400);
//
//	genComplexScene << <1, 1 >> > (scene);
//	CHK_CUDA(cudaGetLastError());
//	CHK_CUDA(cudaDeviceSynchronize());
//	return scene;
//}




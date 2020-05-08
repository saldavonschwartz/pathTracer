//
//  Scene.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

//#include "Scene.hpp"
//#include "Sphere.hpp"
//#include "Material.hpp"
//#include "Utils.hpp"
//#include "HittableVector.hpp"
//#include "Camera.hpp"
//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "curand_kernel.h"


//__global__ void genSimpleScene(Hittable** data, Hittable** scene, Camera* cam, float aspect) {
//	data[0] = new Sphere(gvec3(0.f, 0.f, -1.f), 0.5f, new Diffuse(gvec3(0.1f, 0.2f, 0.5f)));
//	data[1] = new Sphere(gvec3(0.f, -100.5f, -1.f), 100.f, new Diffuse(gvec3(0.8f, 0.8f, 0.0f)));
//	data[2] = new Sphere(gvec3(1.f, 0.f, -1.f), 0.5f, new Metal(gvec3(0.8f, 0.6f, 0.2f), 0.3f));
//	data[3] = new Sphere(gvec3(-1.f, 0.f, -1.f), 0.5f, new Dielectric(1.5f));
//	data[4] = new Sphere(gvec3(-1.f, 0.f, -1.f), -0.45f, new Dielectric(1.5f));
//	*scene = new HittableVector();
//	((HittableVector*)*scene)->data = data;
//	((HittableVector*)*scene)->size = 5;
//	/*scene->init(5);
//
//	scene->add(new Sphere(gvec3(0.f, 0.f, -1.f), 0.5f, new Diffuse(gvec3(0.1f, 0.2f, 0.5f))));
//	scene->add(new Sphere(gvec3(0.f, -100.5f, -1.f), 100.f, new Diffuse(gvec3(0.8f, 0.8f, 0.0f))));
//	scene->add(new Sphere(gvec3(1.f, 0.f, -1.f), 0.5f, new Metal(gvec3(0.8f, 0.6f, 0.2f), 0.3f)));
//	scene->add(new Sphere(gvec3(-1.f, 0.f, -1.f), 0.5f, new Dielectric(1.5f)));
//	scene->add(new Sphere(gvec3(-1.f, 0.f, -1.f), -0.45f, new Dielectric(1.5f)));
//*/
//	gvec3 lookFrom = { 3.f, 3.f, 2.f };
//	gvec3 lookAt = { 0.f, 0.f, -1.f };
//	float f = length(lookFrom - lookAt);
//	float a = 0.1f;
//	float fovy = 20.f;
//	cam->init(lookFrom, lookAt, fovy, aspect, f, a);
//}
//
//void generateSimpleScene(float aspect, Hittable*** scene, Camera** cam) {
//	Hittable** data;
//	CHK_CUDA(cudaMalloc(&data, sizeof(Hittable*) * 5));
//	CHK_CUDA(cudaMalloc(&*scene, sizeof(Hittable*)));
//	CHK_CUDA(cudaMalloc(&*cam, sizeof(Camera)));
//
//	genSimpleScene << <1, 1 >> > (data, *scene, *cam, aspect);
//	CHK_CUDA(cudaGetLastError());
//	CHK_CUDA(cudaDeviceSynchronize());
//}


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




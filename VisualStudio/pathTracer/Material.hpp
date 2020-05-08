//
//  Material.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef Material_hpp
#define Material_hpp

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Ray.hpp"
#include "Hittable.hpp"
#include "Utils.hpp"

using HitInfo = Hittable::HitInfo;


class Material {
public:
  __device__ virtual bool scatter(const Ray& ray, const HitInfo& info, gvec3& attenuation, Ray& scattered, curandState* rs) const = 0;
	__device__ virtual ~Material() {};
};

class Diffuse : public Material {
public:
  gvec3 albedo;
  
	__device__ Diffuse(const gvec3& albedo) 
		: albedo(albedo) {}

	__device__ bool scatter(const Ray& ray, const HitInfo& info, gvec3& attenuation, Ray& scattered, curandState* rs) const override {
		gvec3 scatterDir = info.normal + sphericalRand(1.f, rs);
		scattered = Ray(info.hitPoint, scatterDir);
		attenuation = albedo;
		return true;
	}
};

class Metal : public Material {
public:
  gvec3 albedo;
  float fuzziness;
  
	__device__ Metal(const gvec3& albedo, float fuzziness) 
		: albedo(albedo), fuzziness(fuzziness) {}
	
	__device__ bool scatter(const Ray& ray, const HitInfo& info, gvec3& attenuation, Ray& scattered, curandState* rs) const override {
		gvec3 scatterDir = reflect(ray.dir, info.normal);
		scattered = Ray(info.hitPoint, scatterDir + fuzziness * ballRand(1.f, rs));
		attenuation = albedo;
		return dot(scattered.dir, info.normal) > 0.f;
	}
};

class Dielectric : public Material {
public:
  float refractionIdx;
  
	__device__ Dielectric(float refractionIdx) 
		: refractionIdx(refractionIdx) {}

	__device__ bool scatter(const Ray& ray, const HitInfo& info, gvec3& attenuation, Ray& scattered, curandState* rState) const override {
		attenuation = gvec3{ 1 };
		auto k1 = info.isFrontFace ? 1.f : refractionIdx;
		auto k2 = info.isFrontFace ? refractionIdx : 1.f;
		auto r = reflectance(ray.dir, info.normal, k1, k2);

		if (r == 1.f || curand_uniform(rState) < r) {
			gvec3 scatterDir = reflect(ray.dir, info.normal);
			scattered = Ray(info.hitPoint, scatterDir);
			return true;
		}

		gvec3 scatterDir = refract(ray.dir, info.normal, k1, k2);
		scattered = Ray(info.hitPoint, scatterDir);
		return true;
	}
};

#endif /* Material_hpp */

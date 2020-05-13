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
#include "Ray.hpp"
#include "Hittable.hpp"
#include "Utils.hpp"

using HitInfo = Hittable::HitInfo;


class Material {
public:

  __device__ virtual ~Material() {};
  __device__ virtual gvec3 emission() { return { 0.f }; }
  __device__ virtual bool scatter(const Ray& ray, const HitInfo& info, gvec3& attenuation, Ray& scattered, curandState* rs) const = 0;
};

class Diffuse : public Material {
public:
  gvec3 albedo;
  
	__device__ Diffuse(const gvec3& albedo);
	__device__ bool scatter(const Ray& ray, const HitInfo& info, gvec3& attenuation, Ray& scattered, curandState* rs) const override;
};

class Metal : public Material {
public:
  gvec3 albedo;
  float fuzziness;
  
	__device__ Metal(const gvec3& albedo, float fuzziness);		
	__device__ bool scatter(const Ray& ray, const HitInfo& info, gvec3& attenuation, Ray& scattered, curandState* rs) const override;
};

class Dielectric : public Material {
public:
  float refractionIdx;
  
	__device__ Dielectric(float refractionIdx);
	__device__ bool scatter(const Ray& ray, const HitInfo& info, gvec3& attenuation, Ray& scattered, curandState* rState) const override;
};

class AreaLight : public Material {
public:
  gvec3 color;

  __device__ AreaLight(const gvec3& color);
  __device__ gvec3 emission() override;
  __device__ bool scatter(const Ray& ray, const HitInfo& info, gvec3& attenuation, Ray& scattered, curandState* rState) const override;
};

#endif /* Material_hpp */

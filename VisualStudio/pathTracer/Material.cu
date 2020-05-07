//
//  Material.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "Material.hpp"
#include "Utils.hpp"

__device__ bool Diffuse::scatter(const Ray& ray, const HitInfo& info, gvec3& attenuation, Ray& scattered, curandState* rState) const {
  gvec3 scatterDir = info.normal + sphericalRand(1.f, rState);
  scattered = Ray(info.hitPoint, scatterDir);
  attenuation = albedo;
  return true;
}

__device__ bool Metal::scatter(const Ray& ray, const HitInfo& info, gvec3& attenuation, Ray& scattered, curandState* rState) const {
  gvec3 scatterDir = reflect(ray.dir, info.normal);
  scattered = Ray(info.hitPoint, scatterDir + fuzziness * ballRand(1.f, rState));
  attenuation = albedo;
  return dot(scattered.dir, info.normal) > 0.f;
}

__device__ bool Dielectric::scatter(const Ray& ray, const HitInfo& info, gvec3& attenuation, Ray& scattered, curandState* rState) const {
  attenuation = gvec3{1};
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

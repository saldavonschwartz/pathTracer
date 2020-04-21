//
//  Material.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "Material.hpp"

#include <glm/gtc/random.hpp>
#include "Utils.hpp"

bool Diffuse::scatter(const Ray& ray, const HitInfo& info, vec3& attenuation, Ray& scattered) const {
  vec3 scatterDir = info.normal + glm::sphericalRand(1.f);
  scattered = Ray(info.hitPoint, scatterDir);
  attenuation = albedo;
  return true;
}

bool Metal::scatter(const Ray& ray, const HitInfo& info, vec3& attenuation, Ray& scattered) const {
  vec3 scatterDir = reflect(ray.dir, info.normal);
  scattered = Ray(info.hitPoint, scatterDir + fuzziness * glm::ballRand(1.f));
  attenuation = albedo;
  return dot(scattered.dir, info.normal) > 0.f;
}

bool Dielectric::scatter(const Ray& ray, const HitInfo& info, vec3& attenuation, Ray& scattered) const {
  attenuation = vec3{1};
  auto k1 = info.isFrontFace ? 1.f : refractionIdx;
  auto k2 = info.isFrontFace ? refractionIdx : 1.f;
  auto r = reflectance(ray.dir, info.normal, k1, k2);
  
  if (r == 1.f || urand(0.f, 1.f) < r) {
    vec3 scatterDir = reflect(ray.dir, info.normal);
    scattered = Ray(info.hitPoint, scatterDir);
    return true;
  }
  
  vec3 scatterDir = refract(ray.dir, info.normal, k1, k2);
  scattered = Ray(info.hitPoint, scatterDir);
  return true;
}

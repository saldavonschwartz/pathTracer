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

using HitInfo = Hittable::HitInfo;

class Material {
public:
  virtual bool scatter(const Ray& ray, const HitInfo& info, vec3& attenuation, Ray& scattered) const = 0;
  virtual ~Material() = default;
};

class Diffuse : public Material {
public:
  vec3 albedo;
  
  Diffuse(const vec3& albedo)
  : albedo(albedo) {}
  
  bool scatter(const Ray& ray, const HitInfo& info, vec3& attenuation, Ray& scattered) const override;
};

class Metal : public Material {
public:
  vec3 albedo;
  float fuzziness;
  
  Metal(const vec3& albedo, float fuzziness)
  : albedo(albedo), fuzziness(fuzziness) {}
  
  bool scatter(const Ray& ray, const HitInfo& info, vec3& attenuation, Ray& scattered) const override;
};

class Dielectric : public Material {
public:
  float refractionIdx;
  
  Dielectric(float refractionIdx)
  : refractionIdx(refractionIdx) {}
  
  bool scatter(const Ray& ray, const HitInfo& info, vec3& attenuation, Ray& scattered) const override;
};

#endif /* Material_hpp */

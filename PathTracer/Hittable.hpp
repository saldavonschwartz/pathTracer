//
//  Hittable.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef Hittable_hpp
#define Hittable_hpp

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <memory>
#include "Ray.hpp"

using glm::vec3;
using glm::vec4;

class Material;

class AABA {
public:
  vec3 xmin{0.f};
  vec3 xmax{0.f};
  
  static AABA surroundingBox(const AABA& bBox0, const AABA& bBox1);
  
  AABA() = default;
  
  AABA(vec3 xmin, vec3 xmax)
  : xmin(xmin), xmax(xmax) {}
  
  bool hit(const Ray& ray, float tmin, float tmax) const;
};

class Hittable {
public:
  struct HitInfo {
    vec3 hitPoint{0.f};
    vec3 normal{0.f};
    bool isFrontFace = false;
    float t = 0;
    std::shared_ptr<Material> material;
  };
  
  virtual bool hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const = 0;
  virtual bool boundingBox(double t0, double t1, AABA& bBox) const = 0;
  virtual ~Hittable() = default;
};


#endif /* Hittable_hpp */

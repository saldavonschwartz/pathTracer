//
//  Sphere.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef Sphere_hpp
#define Sphere_hpp

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Hittable.hpp"

class Sphere : public Hittable {
public:
  std::shared_ptr<Material> material;
  float radius;
  vec3 position;
  
  Sphere(vec3 position, float radius, std::shared_ptr<Material> material)
  : position(position), radius(radius), material(material) {}
  
  bool boundingBox(double t0, double t1, AABA& bBox) const override;
  bool hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const override;
};



#endif /* Sphere_hpp */

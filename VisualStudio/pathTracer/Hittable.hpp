//
//  Hittable.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef Hittable_hpp
#define Hittable_hpp

#include "cuda_runtime.h"
#include "Ray.hpp"

class Material;

class AABA {
public:
  gvec3 xmin;
  gvec3 xmax;
  
  __device__ static AABA surroundingBox(const AABA& bBox0, const AABA& bBox1);
	__device__ AABA();
	__device__ AABA(gvec3 xmin, gvec3 xmax);
	__device__ bool hit(const Ray& ray, float tmin, float tmax) const;
};

class Hittable {
public:
  struct HitInfo {
    gvec3 hitPoint{0.f};
    gvec3 normal{0.f};
    bool isFrontFace = false;
    float t = 0;
    Material* material;
  };
  
  __device__ virtual bool hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const = 0;
  __device__ virtual bool boundingBox(double t0, double t1, AABA& bBox) const = 0;
  __device__ virtual ~Hittable() {};
};


#endif /* Hittable_hpp */

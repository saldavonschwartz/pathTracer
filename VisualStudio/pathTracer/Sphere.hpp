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
#include "Hittable.hpp"

class Sphere : public Hittable {
public:
  Material* material;
	gvec3 position;
	float radius;
  
	__device__ Sphere(gvec3 position, float radius, Material* material);
	__device__ ~Sphere() override;
	__device__ bool boundingBox(double t0, double t1, AABA& bBox) const override;
	__device__ bool hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const override;
};



#endif /* Sphere_hpp */

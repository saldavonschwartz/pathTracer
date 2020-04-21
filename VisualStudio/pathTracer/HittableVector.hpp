//
//  HittableVector.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef HittableVector_hpp
#define HittableVector_hpp

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Hittable.hpp"

class HittableVector : public Hittable {
public:
	Hittable** data = nullptr;
	size_t size = 0;

	__host__ __device__ HittableVector() = default;
	__host__ __device__ HittableVector(Hittable** data, size_t size) : data(data), size(size) {}
  //__host__ __device__ void push_back(Hittable* hittable);
  __host__ __device__ bool boundingBox(double t0, double t1, AABA& bBox) const override;
  __host__ __device__ bool hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const override;
};


#endif /* HittableVector_hpp */

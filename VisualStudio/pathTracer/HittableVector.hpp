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
#include "Hittable.hpp"

class HittableVector : public Hittable {
public:
	Hittable** data = nullptr;
	size_t size = 0;
	size_t capacity = 0;
	
	__device__ HittableVector();
	__device__ ~HittableVector() override;
	__device__ void init(size_t capacity);
	__device__ void add(Hittable* hittable);
	__device__ bool boundingBox(double t0, double t1, AABA& bBox) const override;
	__device__ bool hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const override;
};


#endif /* HittableVector_hpp */

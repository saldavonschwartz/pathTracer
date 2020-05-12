//
//  BVH.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef BVH_hpp
#define BVH_hpp

#include "Hittable.hpp"

class HittableVector;

class BVHNode : public Hittable {
public:
	Hittable* right{nullptr};
	Hittable* left{nullptr};
  AABA bBox;

	__device__ BVHNode() {};
	__device__ void sillySort(Hittable** data, int s, int e, int axis);
	__device__ BVHNode(HittableVector* objects, float t0, float t1, curandState* rs);
	__device__ void init(Hittable** objects, int start, int end, float t0, float t1, curandState* rs);
	__device__ bool boundingBox(double t0, double t1, AABA& bBox) const override;
	__device__ bool hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const override;
	__device__ ~BVHNode() override;
};



#endif /* BVH_hpp */

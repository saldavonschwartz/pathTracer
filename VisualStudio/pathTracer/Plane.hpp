//
//  Plane.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef Plane_hpp
#define Plane_hpp

#include "Hittable.hpp"

class Plane : public Hittable {
public:
	Material* material;
	float width, height;
	gvec3 position;

	__device__ Plane(const gvec3& position, float width, float height, Material* material);
	__device__ ~Plane() override;
	__device__ bool boundingBox(double t0, double t1, AABA& bBox) const override;
	__device__ bool hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const override;
};

#endif /* BVH_hpp */


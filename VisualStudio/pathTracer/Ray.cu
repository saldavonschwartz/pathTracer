//
//  Ray.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "Ray.hpp"

__device__  Ray::Ray(const gvec3& origin, const gvec3& dir)
	: origin(origin), dir(normalize(dir)) {}

__device__ gvec3 Ray::operator()(float t) const {
	return origin + t * dir;
}
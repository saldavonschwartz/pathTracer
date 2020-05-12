//
//  Ray.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef Ray_hpp
#define Ray_hpp

#include "Utils.hpp"

class Ray {
public:
	gvec3 origin = {0.f};
	gvec3 dir = {0.f};

	__device__  Ray() {};
	__device__  Ray(const gvec3& origin, const gvec3& dir);
	__device__ gvec3 operator()(float t) const;
};


#endif /* Ray_hpp */

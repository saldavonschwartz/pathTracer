//
//  Hittable.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "Hittable.hpp"

//__device__ AABA AABA::surroundingBox(const AABA& b0, const AABA& b1) {
//	gvec3 small = { 
//		fminf(b0.xmin.x, b1.xmin.x), 
//		fminf(b0.xmin.y, b1.xmin.y) , 
//		fminf(b0.xmin.z, b1.xmin.z) 
//	};
//	
//	gvec3 big = {
//		fmaxf(b0.xmax.x, b1.xmax.x), 
//		fmaxf(b0.xmax.y, b1.xmax.y), 
//		fmaxf(b0.xmax.z, b1.xmax.z)
//	};
//
//	return {small, big};
//}

__device__ bool AABA::hit(const Ray& ray, float tmin, float tmax) const {
	// TODO: 0xfede
  /*auto xminIntersect = (xmin - ray.origin) / ray.dir;
  auto xmaxIntersect = (xmax - ray.origin) / ray.dir;
  auto t0 = min(xminIntersect, xmaxIntersect);
  auto t1 = max(xminIntersect, xmaxIntersect);
  tmin = compMax(gvec4(t0, tmin));
  tmax = compMin(gvec4(t1, tmax));
  return tmin < tmax;*/

	for (int a = 0; a < 3; a++) {
		auto invD = 1.0f / ray.dir[a];
		auto t0 = (xmin[a] - ray.origin[a]) * invD;
		auto t1 = (xmax[a] - ray.origin[a]) * invD;
		
		if (invD < 0.0f) {
			auto tn = t0;
			t0 = t1;
			t1 = tn;
		}

		tmin = t0 > tmin ? t0 : tmin;
		tmax = t1 < tmax ? t1 : tmax;
		
		if (tmax <= tmin) {
			return false;
		}
	}

	return true;
}

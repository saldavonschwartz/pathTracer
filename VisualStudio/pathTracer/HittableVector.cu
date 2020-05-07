//
//  HittableVector.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "HittableVector.hpp"

__device__ void HittableVector::push_back(Hittable* hittable) {
	data[idx] = hittable;
	idx = (idx + 1) % size;
}

__device__ bool HittableVector::boundingBox(double t0, double t1, AABA& bBox) const {
  bool firstTime = true;
  AABA bBoxTemp;
  
  for (size_t i = 0; i < size; i++) {
	  auto h = data[i];

    if (!h->boundingBox(t0, t1, bBoxTemp)) {
      return false;
    }
    
    if (firstTime) {
      firstTime = false;
      bBox = bBoxTemp;
    }
    else {
      bBox = AABA::surroundingBox(bBox, bBoxTemp);
    }
  }
  
  return true;
}

__device__ bool HittableVector::hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const {
  float closest = tmax;
  bool hitAny = false;
  HitInfo _info;
  
	for (size_t i = 0; i < size; i++) {
		auto h = data[i];

    if (h->hit(ray, tmin, closest, _info)) {
      closest = _info.t;
      hitAny = true;
    }
  }
  
  if (hitAny) {
    info = _info;
  }
  
  return hitAny;
}

__device__ HittableVector::~HittableVector() {
	for (int i = 0; i < size; i++) {
		delete data[i];
	}

	size = idx = 0;
}
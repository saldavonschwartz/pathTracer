//
//  HittableVector.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "HittableVector.hpp"

void HittableVector::push_back(shared_ptr<Hittable> hittable) {
  data.push_back(hittable);
}

bool HittableVector::boundingBox(double t0, double t1, AABA& bBox) const {
  bool firstTime = true;
  AABA bBoxTemp;
  
  for (auto& h: data) {
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

bool HittableVector::hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const {
  float closest = tmax;
  bool hitAny = false;
  HitInfo _info;
  
  for (auto& h : data) {
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


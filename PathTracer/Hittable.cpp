//
//  Hittable.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "Hittable.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/component_wise.hpp>

AABA AABA::surroundingBox(const AABA& bBox0, const AABA& bBox1) {
  return AABA(min(bBox0.xmin, bBox1.xmin), max(bBox0.xmax, bBox1.xmax));
}

bool AABA::hit(const Ray& ray, float tmin, float tmax) const {
  auto xminIntersect = (xmin - ray.origin) / ray.dir;
  auto xmaxIntersect = (xmax - ray.origin) / ray.dir;
  auto t0 = min(xminIntersect, xmaxIntersect);
  auto t1 = max(xminIntersect, xmaxIntersect);
  tmin = compMax(vec4(t0, tmin));
  tmax = compMin(vec4(t1, tmax));
  return tmin < tmax;
}

//
//  Camera.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef Camera_hpp
#define Camera_hpp

#include "cuda_runtime.h"
#include "Ray.hpp"
#include "Utils.hpp"

class Camera {
public:
  gvec3 position = {0.f};
  float aspect = 0.f;
  float f = 0.f;
  float a = 0.f;
  float fovy = 0.f;

	__device__ void init(const gvec3& position, const gvec3& lookAt, float fovy, float aspect, float flLength, float aperture);
	__device__ Ray castRay(float u, float v, curandState* rs) const;

private:
  float hh, hw;
  gvec3 x, y;
  gvec3 hOffset;
  gvec3 wOffset;
  gvec3 lowerLeftImageOrigin;
};

#endif /* Camera_hpp */

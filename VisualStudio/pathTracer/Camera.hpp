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
#include "device_launch_parameters.h"
#include "Ray.hpp"
#include "Utils.hpp"

class Camera {
public:
  gvec3 position;
  float aspect;
  float f;
  float a;
  float fovy;

	__device__ void init(
		const gvec3& position, const gvec3& lookAt, float fovy, 
		float aspect, float flLength, float aperture
	) 
	{
		this->position = position;
		this->aspect = aspect;
		this->f = flLength;
		this->a = aperture;
		this->fovy = fovy;

		float hh = tanf((fovy * (pi / 180.f) / 2.f));
		float hw = aspect * hh;

		// [x y z p] = new camera orientation:
		gvec3 z = normalize(position - lookAt);
		x = normalize(cross({ 0.f, 1.f, 0.f }, z));
		y = cross(z, x);

		auto& f = flLength;
		lowerLeftImageOrigin = position - x * hw*f - y * hh*f - z * f;
		hOffset = 2 * hh*f*y;
		wOffset = 2 * hw*f*x;
	}

	__device__ Ray castRay(float u, float v, curandState* rs) const {
		auto r = (a / 2.f) * diskRand(1.f, rs);
		auto offset = x * r.x + y * r.y;
		auto dir = lowerLeftImageOrigin + u * wOffset + v * hOffset - position - offset;
		return { position + offset, dir };
	}

private:
  float hh, hw;
  gvec3 x, y;
  gvec3 hOffset;
  gvec3 wOffset;
  gvec3 lowerLeftImageOrigin;
};

#endif /* Camera_hpp */

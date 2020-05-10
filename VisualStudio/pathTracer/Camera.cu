//
//  Camera.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "Camera.hpp"
#include "Utils.hpp"


__device__ Ray Camera::castRay(float u, float v, curandState* rs) const {
	auto r = (a / 2.f) * diskRand(1.f, rs);
	auto offset = x * r.x + y * r.y;
	auto dir = lowerLeftImageOrigin + u * wOffset + v * hOffset - position - offset;
	return { position + offset, dir };
}

__device__ void Camera::init(
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

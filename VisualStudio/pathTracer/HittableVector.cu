//
//  HittableVector.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "HittableVector.hpp"

__device__ void HittableVector::init(int capacity) {
	this->capacity = capacity;
	this->size = 0;
	data = new Hittable*[capacity];
}

__device__ HittableVector::~HittableVector() {
	delete[] data;
	size = capacity = 0;
}

__device__ void HittableVector::add(Hittable* hittable) {
	data[size++] = hittable;
}

__device__ bool HittableVector::boundingBox(double t0, double t1, AABA& bBox) const {
	bool firstTime = true;
	AABA bBoxTemp;

	for (int i = 0; i < size; i++) {
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

	for (int i = 0; i < size; i++) {
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

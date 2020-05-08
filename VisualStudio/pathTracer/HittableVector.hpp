//
//  HittableVector.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef HittableVector_hpp
#define HittableVector_hpp

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Hittable.hpp"

class HittableVector : public Hittable {
public:
	Hittable** data = nullptr;
	size_t size = 0;
	size_t idx = 0;

	__device__ HittableVector() {}

	__device__ void init(size_t size) {
		this->size = size;
		this->data = new Hittable*[size];
	}
  
	__device__ void HittableVector::add(Hittable* hittable) {
		data[idx] = hittable;
		idx = (idx + 1) % size;
	}

	__device__ bool boundingBox(double t0, double t1, AABA& bBox) const override {
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
				bBox = surroundingBox(bBox, bBoxTemp);
			}
		}

		return true;
	}

	__device__ bool hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const override {
		float closest = tmax;
		bool hitAny = false;
		HitInfo _info;

		for (size_t i = 0; i < size; i++) {
			auto h = data[i];

			if (!h) {
				continue;
			}

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

	__device__ HittableVector::~HittableVector() override {
		delete[] data;
		size = idx = 0;
	}
};


#endif /* HittableVector_hpp */

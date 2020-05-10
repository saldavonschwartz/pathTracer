//
//  Sphere.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "Sphere.hpp"
#include "Material.hpp"

__device__ Sphere::Sphere(gvec3 position, float radius, Material* material)
	: position(position), radius(radius), material(material) {}

__device__ Sphere::~Sphere() {
	delete material;
}

__device__ bool Sphere::boundingBox(double t0, double t1, AABA& bBox) const {
	gvec3 extent{ radius };
	bBox = AABA(position - extent, position + extent);
	return true;
}

__device__ bool Sphere::hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const {
	// From ray eq. r(t) = 0 + t*d and sphere eq. (x-c)^2 - r^2 = 0
	// Solve quadratic (r(t)-c)^2 -r^2 = 0 -> a*t^2 + b*t + c = 0
	// 0 roots = no hit, 1 root = tanget hit, 2 roots = went in and thru:

	gvec3 oc = ray.origin - position;
	float a = dot(ray.dir, ray.dir);
	float b = dot(oc, ray.dir);
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;

	if (discriminant > 0) {
		float t = (-b - sqrt(discriminant)) / a;

		if (tmin < t && t < tmax) {
			auto hitPoint = ray(t);
			info.hitPoint = hitPoint;
			info.normal = (hitPoint - position) / radius;
			info.isFrontFace = dot(info.normal, ray.dir) < 0.f;
			info.normal = info.isFrontFace ? info.normal : -info.normal;
			info.t = t;
			info.material = material;
			return true;
		}

		t = (-b + sqrt(discriminant)) / a;

		if (tmin < t && t < tmax) {
			auto hitPoint = ray(t);
			info.hitPoint = hitPoint;
			info.normal = (hitPoint - position) / radius;
			info.isFrontFace = dot(info.normal, ray.dir) < 0.f;
			info.normal = info.isFrontFace ? info.normal : -info.normal;
			info.t = t;
			info.material = material;
			return true;
		}
	}

	return false;
}
//
//  BVH.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef BVH_hpp
#define BVH_hpp

#include "Hittable.hpp"
#include "HittableVector.hpp"
#include <nvfunctional>

class BVHNode : public Hittable {
public:
  Hittable* right;
  Hittable* left;
  AABA bBox;

	__device__ void sillySort(
		Hittable** data, int s, int e, 
		nvstd::function<bool((const Hittable*, const Hittable*))> cmp) {
		for (int i = s; i < e; i++) {
			int j = i;

			for (int k = i+1; k < e; k++) {
				if (cmp(data[k], data[j])) {
					j = k;
				}
			}

			Hittable* temp = data[i];
			data[i] = data[j];
			data[j] = temp;
		}
	}

  __device__ BVHNode(HittableVector list, float t0, float t1, curandState* rs)
  : BVHNode(list.data, 0, list.size, t0, t1, rs) {}
  
	__device__ BVHNode(Hittable** objects, size_t start, size_t end, float t0, float t1, curandState* rs) {
		int axis = ceilf(curand_uniform(rs) * 3);
		auto cmp = [axis](const Hittable* h1, const Hittable* h2) {
			AABA bBox1;
			AABA bBox2;
    
			if (!h1->boundingBox(0,0, bBox1) || !h2->boundingBox(0,0, bBox2)) {
				printf("No bounding box in BVHNode constructor.\n");
			}
    
			return bBox1.xmin[axis] < bBox2.xmin[axis];
		};
  
		auto objectsLeft = end - start;
  
		switch (objectsLeft) {
			case 1: {
				left = right = objects[start];
				break;
			}
      
			case 2: {
				if (cmp(objects[start], objects[start+1])) {
					left = objects[start];
					right = objects[start+1];
				}
				else {
					left = objects[start+1];
					right = objects[start];
				}
      
				break;
			}
      
			default: {
				sillySort(objects, start, end, cmp);
				//sort(objects.begin() + start, objects.begin() + end, cmp);
				auto mid = start + objectsLeft/2;
				left = new BVHNode(objects, start, mid, t0, t1, rs);
				right = new BVHNode(objects, mid, end, t0, t1, rs);
      
				break;
			}
		}
  
		AABA bBoxLeft, bBoxRight;
  
		if (!left->boundingBox(t0, t1, bBoxLeft) || !right->boundingBox(t0, t1, bBoxRight)) {
			printf("No bounding box in BVHNode constructor.\n");
		}
  
		bBox = surroundingBox(bBoxLeft, bBoxRight);
	}

	__device__ bool BVHNode::boundingBox(double t0, double t1, AABA& bBox) const override {
		bBox = this->bBox;
		return true;
	}

	__device__ bool BVHNode::hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const override {
		if (!bBox.hit(ray, tmin, tmax)) {
			return false;
		}
  
		bool leftHit = left->hit(ray, tmin, tmax, info);
		bool rightHit = right->hit(ray, tmin,  leftHit ? info.t : tmax, info);
		return leftHit || rightHit;
	}
};



#endif /* BVH_hpp */

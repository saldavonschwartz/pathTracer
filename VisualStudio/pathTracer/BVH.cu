//
//  BVH.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "BVH.hpp"
#include "HittableVector.hpp"

__device__ bool cmp(int axis, Hittable* h1, Hittable* h2) {
	AABA bBox1;
	AABA bBox2;

	if (!h1->boundingBox(0, 0, bBox1) || !h2->boundingBox(0, 0, bBox2)) {
		printf("No bounding box in BVHNode constructor.\n");
	}

	return bBox1.xmin[axis] < bBox2.xmin[axis];
}

__device__ void BVHNode::sillySort(Hittable** data, int s, int e, int axis) {
	for (int i = s; i < e; i++) {
		int j = i;

		for (int k = i + 1; k < e; k++) {
			if (cmp(axis, data[k], data[j])) {
				j = k;
			}
		}

		Hittable* temp = data[i];
		data[i] = data[j];
		data[j] = temp;
	}
}

__device__ BVHNode::BVHNode(HittableVector* objects, float t0, float t1, curandState* rs) {
	init(objects->data, 0, objects->size - 1, t0, t1, rs);
	/*int axis = ceilf(curand_uniform(rs) * 3);
	int objectsLeft = end - start;
	
	switch (objectsLeft) {
		case 1: {
			left = right = objects[start];
			break;
		}

		case 2: {
			if (cmp(axis, objects[start], objects[start + 1])) {
				left = objects[start];
				right = objects[start + 1];
			}
			else {
				left = objects[start + 1];
				right = objects[start];
			}
			break;
		}

		default: {
			sillySort(objects, start, end, axis);
			int mid = start + objectsLeft / 2;
			left = new BVHNode(objects, start, mid, t0, t1, rs);
			right = new BVHNode(objects, mid, end, t0, t1, rs);
			break;
		}
	}

	AABA bBoxLeft, bBoxRight;

	if (!left->boundingBox(t0, t1, bBoxLeft) || !right->boundingBox(t0, t1, bBoxRight)) {
		printf("No bounding box in BVHNode constructor.\n");
	}

	bBox = AABA::surroundingBox(bBoxLeft, bBoxRight);*/
}

struct NodeInfo {
	size_t s, e;
	BVHNode* bvh;
};

__device__ void BVHNode::init(Hittable** objects, int start, int end, float t0, float t1, curandState* rs) {
	auto next = [this, objects, rs] __device__ (NodeInfo n, NodeInfo * stack, int&  i) {
		bool done = false;

		while (!done) {
			int objectsLeft = n.e - n.s;

			if (objectsLeft > 2) {
				int axis = ceilf(curand_uniform(rs) * 3);
				sillySort(objects, n.s, n.e, axis);
				int mid = n.s + objectsLeft / 2;
				n.bvh->right = new BVHNode();
				n.bvh->left = new BVHNode();
				stack[++i] = { mid, n.e, ((BVHNode*)n.bvh->right) }; 
				stack[++i] = n;
				n = { n.s, mid, ((BVHNode*)n.bvh->left) };
			}
			else {
				stack[++i] = n;
				done = true;
			}
		}
	};

	NodeInfo* stack = new NodeInfo[end * 2];
	int i = -1;
	next({ start, end, this }, stack, i);

	while (i >= 0) {
		auto n = stack[i--];
		
		if (i >= 0 && stack[i].bvh == n.bvh->right) {
			auto nRight = stack[i];
			stack[i] = n;
			next(nRight, stack, i);
			n = stack[i--];
		}

		int objectsLeft = n.e - n.s;

		switch (objectsLeft) {
		case 1: {
			n.bvh->left = n.bvh->right = objects[n.s];
			break;
		}

		case 2: {
			int axis = ceilf(curand_uniform(rs) * 3);
			
			if (cmp(axis, objects[n.s], objects[n.s + 1])) {
				n.bvh->left = objects[n.s];
				n.bvh->right = objects[n.s + 1];
			}
			else {
				n.bvh->left = objects[n.s + 1];
				n.bvh->right = objects[n.s];
			}

			break;
		}
		}

		AABA bBoxLeft, bBoxRight;

		if (!n.bvh->left->boundingBox(t0, t1, bBoxLeft) || !n.bvh->right->boundingBox(t0, t1, bBoxRight)) {
			printf("No bounding box in BVHNode constructor.\n");
		}

		n.bvh->bBox = AABA::surroundingBox(bBoxLeft, bBoxRight);
	}
}

__device__ BVHNode::~BVHNode() {
	if (left != right) {
		delete left;
		delete right;
	}
	else {
		delete left;
	}
}

__device__ bool BVHNode::boundingBox(double t0, double t1, AABA& bBox) const {
	bBox = this->bBox;
	return true;
}

__device__ bool BVHNode::hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const {
	if (!bBox.hit(ray, tmin, tmax)) {
		return false;
	}

	bool leftHit = left->hit(ray, tmin, tmax, info);
	bool rightHit = right->hit(ray, tmin, leftHit ? info.t : tmax, info);
	return leftHit || rightHit;
}
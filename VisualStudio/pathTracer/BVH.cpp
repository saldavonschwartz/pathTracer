//
//  BVH.cpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#include "BVH.hpp"
#include <iostream>
#include <algorithm>

using std::cerr;
using std::make_shared;
using std::sort;

BVHNode::BVHNode(std::vector<shared_ptr<Hittable>>& objects, size_t start, size_t end, float t0, float t1) {
  int axis = rand() % 3;
  auto cmp = [axis](const shared_ptr<Hittable> h1, const shared_ptr<Hittable> h2){
    AABA bBox1;
    AABA bBox2;
    
    if (!h1->boundingBox(0,0, bBox1) || !h2->boundingBox(0,0, bBox2)) {
      cerr << "No bounding box in BVHNode constructor.\n";
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
      sort(objects.begin() + start, objects.begin() + end, cmp);
      auto mid = start + objectsLeft/2;
      left = make_shared<BVHNode>(objects, start, mid, t0, t1);
      right = make_shared<BVHNode>(objects, mid, end, t0, t1);
      
      break;
    }
  }
  
  AABA bBoxLeft, bBoxRight;
  
  if (!left->boundingBox(t0, t1, bBoxLeft) || !right->boundingBox(t0, t1, bBoxRight)) {
    cerr << "No bounding box in BVHNode constructor.\n";
  }
  
  bBox = AABA::surroundingBox(bBoxLeft, bBoxRight);
}

bool BVHNode::boundingBox(double t0, double t1, AABA& bBox) const {
  bBox = this->bBox;
  return true;
}

bool BVHNode::hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const {
  if (!bBox.hit(ray, tmin, tmax)) {
    return false;
  }
  
  bool leftHit = left->hit(ray, tmin, tmax, info);
  bool rightHit = right->hit(ray, tmin,  leftHit ? info.t : tmax, info);
  return leftHit || rightHit;
}

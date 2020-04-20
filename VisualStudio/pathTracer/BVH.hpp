//
//  BVH.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef BVH_hpp
#define BVH_hpp

#include "HittableVector.hpp"

class BVHNode : public Hittable {
public:
  shared_ptr<Hittable> right;
  shared_ptr<Hittable> left;
  AABA bBox;

  BVHNode(HittableVector list, float t0, float t1)
  : BVHNode(list.data, 0, list.data.size(), t0, t1) {}
  
  BVHNode(std::vector<shared_ptr<Hittable>>& objects, size_t start, size_t end, float t0, float t1);

  bool boundingBox(double t0, double t1, AABA& bBox) const override;
  bool hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const override;
};



#endif /* BVH_hpp */

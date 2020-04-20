//
//  HittableVector.hpp
//  PathTracer
//
//  Created by Federico Saldarini on 4/16/20.
//  Copyright © 2020 Federico Saldarini. All rights reserved.
//

#ifndef HittableVector_hpp
#define HittableVector_hpp

#include "Hittable.hpp"

#include <vector>
#include <memory>

using std::vector;
using std::shared_ptr;

class HittableVector : public Hittable {
public:
  vector<shared_ptr<Hittable>> data;
  
  void push_back(shared_ptr<Hittable> hittable);
  bool boundingBox(double t0, double t1, AABA& bBox) const override;
  bool hit(const Ray& ray, float tmin, float tmax, HitInfo& info) const override;
};


#endif /* HittableVector_hpp */

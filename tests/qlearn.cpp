#define BOOST_TEST_MODULE qlearn

#include "QLearning.h"
#include "RefinementTree.h"
#include "propts.h"


#include <boost/test/unit_test.hpp>
#include <random>

using namespace prlearn;

BOOST_AUTO_TEST_CASE(Dummy) {
    BOOST_REQUIRE(true);
}

BOOST_AUTO_TEST_CASE(Linear) {
    std::vector<QLearning<RefinementTree>> learners;
    learners.emplace_back();
    prlearn::propts_t options;
    std::mt19937 e;
    //e.seed((int)time(nullptr));
    std::uniform_real_distribution<> dist(0, 10);
    double points[]{0,0};
    for(size_t i = 0; i < 3000 ; ++i)
    {
        points[0] = dist(e);
        points[1] = dist(e);
        double val = points[0] + points[1];
        learners[0].addSample(2, points, nullptr, nullptr, 1, 0, 0, val, learners, true, 0, options);
        auto v2 = learners[0].lookup(0, points, 2);
    }
    BOOST_REQUIRE(true);
}

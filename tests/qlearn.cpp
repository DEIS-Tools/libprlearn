#define BOOST_TEST_MODULE qlearn

#include "QLearning.h"
#include "RefinementTree.h"
#include "propts.h"


#include <boost/test/unit_test.hpp>
#include <random>
#include <complex>

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
        double val = points[0] + points[1] + 3;
        learners[0].addSample(2, points, nullptr, nullptr, 1, 0, 0, val, learners, true, 0, options);
        auto v2 = learners[0].lookup(0, points, 2);
        if(i > 1000)
        {
            auto diff = std::abs(v2.avg() - val);
            BOOST_REQUIRE_LE(diff, 1);
        }
    }
    BOOST_REQUIRE(true);
}

BOOST_AUTO_TEST_CASE(Linear2) {
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
        double val = points[0]*0.1 + points[1]*10.0 + 3;
        learners[0].addSample(2, points, nullptr, nullptr, 1, 0, 0, val, learners, true, 0, options);
        auto v2 = learners[0].lookup(0, points, 2);
        if(i > 1000)
        {
            auto diff = std::abs(v2.avg() - val);
            BOOST_REQUIRE_LE(diff, 1);
        }
    }
    BOOST_REQUIRE(true);
}


BOOST_AUTO_TEST_CASE(LinearInv) {
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
        points[1] = -dist(e);
        double val = points[0] + points[1] + 3;
        learners[0].addSample(2, points, nullptr, nullptr, 1, 0, 0, val, learners, true, 0, options);
        auto v2 = learners[0].lookup(0, points, 2);
        if(i > 1000)
        {
            auto diff = std::abs(v2.avg() - val);
            BOOST_REQUIRE_LE(diff, 1);
        }
    }
    BOOST_REQUIRE(true);
}

BOOST_AUTO_TEST_CASE(LinearNoise) {
    std::vector<QLearning<RefinementTree>> learners;
    learners.emplace_back();
    prlearn::propts_t options;
    std::mt19937 e;
    //e.seed((int)time(nullptr));
    std::uniform_real_distribution<> dist(0, 10);
    std::uniform_real_distribution<> noise(-1, 1);
    double points[]{0,0};
    for(size_t i = 0; i < 4000 ; ++i)
    {
        points[0] = dist(e);
        points[1] = dist(e);
        double val = points[0] + points[1] + 3;
        learners[0].addSample(2, points, nullptr, nullptr, 1, 0, 0, val + noise(e), learners, true, 0, options);
        auto v2 = learners[0].lookup(0, points, 2);
        if(i > 3000)
        {
            auto diff = std::abs(v2.avg() - val);
            BOOST_REQUIRE_LE(diff, 1);
        }
    }
    BOOST_REQUIRE(true);
}

BOOST_AUTO_TEST_CASE(Unused) {
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
        double val = points[0] + 3;
        learners[0].addSample(2, points, nullptr, nullptr, 1, 0, 0, val, learners, true, 0, options);
        auto v2 = learners[0].lookup(0, points, 2);
        if(i > 1000)
        {
            auto diff = std::abs(v2.avg() - val);
            BOOST_REQUIRE_LE(diff, 1);
        }
    }
    BOOST_REQUIRE(true);
}

BOOST_AUTO_TEST_CASE(Indep2) {
    std::cerr << "InDep2" << std::endl;
    std::vector<QLearning<RefinementTree>> learners;
    learners.emplace_back();
    prlearn::propts_t options;
    std::mt19937 e;
    //e.seed((int)time(nullptr));
    std::uniform_real_distribution<> dist(0, 10);
    double points[]{0,0};
    double od = 0;

    for(size_t i = 0; i < 3000 ; ++i)
    {
        points[0] = dist(e);
        points[1] = 0;
        double val = points[0] + points[1] + 3;
        learners[0].addSample(2, points, nullptr, nullptr, 1, 0, 0, val, learners, true, 0, options);
        auto v2 = learners[0].lookup(0, points, 2);
        if(i > 1000)
        {
            auto diff = std::abs(v2.avg() - val);
            BOOST_REQUIRE_LE(diff, 1);
            if(diff > od)
                std::cerr << "DIFF " << diff << std::endl;
            od = std::max(diff, od);
        }
    }
    BOOST_REQUIRE(true);
}

BOOST_AUTO_TEST_CASE(PartDep) {
    std::cerr << "PartDep" << std::endl;
    std::vector<QLearning<RefinementTree>> learners;
    learners.emplace_back();
    prlearn::propts_t options;
    std::mt19937 e;
    //e.seed((int)time(nullptr));
    std::uniform_real_distribution<> dist(0, 10);
    double points[]{0,0};
    double od = 0;

    for(size_t i = 0; i < 4000 ; ++i)
    {
        points[0] = dist(e);
        points[1] = dist(e) + points[0]*0.5;
        double val = points[0] + points[1] + 3;
        learners[0].addSample(2, points, nullptr, nullptr, 1, 0, 0, val, learners, true, 0, options);
        auto v2 = learners[0].lookup(0, points, 2);
        if(i > 3000)
        {
            auto diff = std::abs(v2.avg() - val);
            BOOST_REQUIRE_LE(diff, 1);
            if(diff > od)
                std::cerr << "DIFF " << diff << std::endl;
            od = std::max(diff, od);
        }
    }
    BOOST_REQUIRE(true);
}

BOOST_AUTO_TEST_CASE(CoDep) {
    std::cerr << "CODEP" << std::endl;
    std::vector<QLearning<RefinementTree>> learners;
    learners.emplace_back();
    prlearn::propts_t options;
    std::mt19937 e;
    //e.seed((int)time(nullptr));
    std::uniform_real_distribution<> dist(0, 10);
   std::uniform_real_distribution<> noise(-1, 1);
    double points[]{0,0};
    double od = 0;

    for(size_t i = 0; i < 10000 ; ++i)
    {
        points[0] = dist(e);
        points[1] = points[0] + noise(e);
        double val = points[0] + points[1] + 3;
        learners[0].addSample(2, points, nullptr, nullptr, 1, 0, 0, val, learners, true, 0, options);
        auto v2 = learners[0].lookup(0, points, 2);
        if(i > 1000)
        {
            auto diff = std::abs(v2.avg() - val);
            BOOST_REQUIRE_LE(diff, 2);
            /*if(diff > 1)
                std::cerr << "DIFF " << diff << std::endl;*/
            od = std::max(diff, od);
        }
    }
    BOOST_REQUIRE(true);
}

BOOST_AUTO_TEST_CASE(Complex) {
    std::cerr << "CODEP" << std::endl;
    std::vector<QLearning<RefinementTree>> learners;
    learners.emplace_back();
    prlearn::propts_t options;
    std::mt19937 e;
    //e.seed((int)time(nullptr));
    std::uniform_real_distribution<> dist(0, 10);
    double points[]{0,0};
    double od = 0;

    for(size_t i = 0; i < 10000 ; ++i)
    {
        points[0] = dist(e);
        points[1] = dist(e);
        double val = std::sin(points[0]) + std::pow(points[1], 2.0);
        learners[0].addSample(2, points, nullptr, nullptr, 1, 0, 0, val, learners, true, 0, options);
        auto v2 = learners[0].lookup(0, points, 2);
        if(i > 1000)
        {
            auto diff = std::abs(v2.avg() - val);
            /*if(diff > 1)
                std::cerr << "DIFF (" << i << ") " << diff << std::endl;*/
            od = std::max(diff, od);
        }
    }
    BOOST_REQUIRE(true);
}
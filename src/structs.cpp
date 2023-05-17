/*
 * Copyright Peter G. Jensen
 *  
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/* 
 * File:   structs.cpp
 * Author: Peter G. Jensen
 *
 * Created on July 25, 2017, 10:31 AM
 */

#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/triangular.hpp>

#include "structs.h"
namespace prlearn {

    std::ostream& operator<<(std::ostream& stream, const avg_t& el) {
        stream << "<" << el._cnt << " : " << el._avg << ">";
        return stream;
    }

    el_t::el_t(size_t l) : _label(l) {
    }

    bool el_t::operator<(const el_t& other) const {
        return _label < other._label;
    }

    void qvar_t::print(std::ostream& stream) const {
        stream << "[";
        stream << (*(avg_t*)this);
        stream << ", " << _variance << "]";
    }

    std::ostream& operator<<(std::ostream& o, const qvar_t& v) {
        v.print(o);
        return o;
    }

    rqvar_t::rqvar_t(qvar_t other) : qvar_t(other.avg(), other.cnt(), other._variance) {
    }

    rqvar_t& rqvar_t::operator+=(double d) {
        assert(!std::isinf(d));
        if(_cnt == 0)
        {
            _avg = d;
            _variance = 0;
            ++_cnt;
        }
        else
        {
            const auto frac = 1.0 / std::max(std::sqrt(_cnt / 5), std::min(3.0, _cnt));//std::max(std::sqrt(_cnt / 10), std::min(3.0, _cnt));
            _avg += (d - _avg) * frac;
            auto nvar = std::pow(d - _avg, 2.0) - _variance;
            ++_cnt;
            const auto vfrac = 1 / _cnt;
            _variance += vfrac * nvar;
        }

        return *this;
    }

    qvar_t qvar_t::approximate(const qvar_t& a, const qvar_t& b) {
        if (a._cnt == 0)
            return b;
        if (b._cnt == 0)
            return a;
        qvar_t res = a;
        res.addPoints(b._cnt, b._avg);
        const auto adif = std::abs(res._avg - a._avg);
        const auto bdif = std::abs(res._avg - b._avg);
        const auto astd = std::sqrt(a._variance);
        const auto bstd = std::sqrt(b._variance);
        auto ca = std::pow(adif + astd, 2.0) + std::pow(adif - astd, 2.0);
        auto cb = std::pow(bdif + bstd, 2.0) + std::pow(bdif - bstd, 2.0);
        avg_t tmp;
        tmp.addPoints(a._cnt, ca / 2.0);
        tmp.addPoints(b._cnt, cb / 2.0);
        res._variance = tmp._avg;
        return res;
    }

    qvar_t& qvar_t::operator+=(double d) {
        assert(!std::isinf(d));
        avg_t::operator+=(d);
        auto nvar = std::pow(d - _avg, 2.0);
        assert(!std::isinf(nvar));
        if (_cnt == 1) _variance = nvar;
        else {
            nvar -= _variance;
            _variance += nvar / _cnt;
        }
        return *this;
    }

    void qvar_t::addPoints(double weight, double d) {
        assert(weight >= 0);
        assert(_cnt >= 0);
        if (weight == 0) return;
        auto oa = _avg;
        avg_t::addPoints(weight, d);
        auto nvar = std::abs((d - oa)*(d - _avg));
        assert(!std::isinf(nvar));
        if (_cnt == weight) _variance = nvar;
        else {
            nvar -= _variance;
            _variance += (nvar * weight) / _cnt;
        }
        assert(_variance >= 0);
        assert(!std::isnan(_variance));
        assert(!std::isinf(_variance));
    }

    double triangular_cdf(double mid, double width, double point) {
        auto cpow = std::pow(width, 2.0);
        if (point < mid)
            return (1.0 / cpow)*((std::pow(point - mid, 2.0) / 2.0) + (std::pow(width, 2.0) / 2.0));
        else
            return 0.5 + ((1.0 / cpow)*((mid * (point - mid)) - (std::pow(point - mid, 2.0) / 2.0)));
    }

    void splitfilter_t::add(const qvar_t& a, const qvar_t& b, double indif, double tl, double tu, double t2, double rate) {
        using namespace boost::math;

        constexpr double minvar = 0.0001;
        if (std::min(a.cnt(), b.cnt()) <= 1)
            return;
        if (a._variance == b._variance && a.avg() == b.avg())
            return;
        auto vara = std::max(minvar, a._variance);
        auto varb = std::max(minvar, b._variance);

        double tval = std::abs(a.avg() - b.avg()) / std::sqrt(((vara * a.cnt()) + (varb * b.cnt())) / (a.cnt() * b.cnt()));

        if (tval >= tu) {
            if (std::abs(a.avg() - b.avg()) < indif)
                return; // don't care, too close
            // t-test approximation heuristic
            _vfilter += (0.0 - _vfilter) * rate;
            double lr = 1.0;
            if (a.avg() > b.avg())
                lr = 0.0;
            _lfilter += (lr - _lfilter) * rate;
            _hfilter += ((1.0 - lr) - _hfilter) * rate;
        } else if (tval <= tl) {
            // KS-approximation heuristic
            auto ca = std::sqrt(-0.5 * std::log(t2 / 2.0));
            auto mes = std::sqrt(((double) (a.cnt() + b.cnt())) / ((double) a.cnt() * b.cnt()));


            auto wa = std::sqrt(vara * 6);
            auto wb = std::sqrt(varb * 6);

            double dist = 0;

            if (std::min(a.avg(), b.avg()) + wa + wb < std::max(a.avg(), b.avg()))
                dist = 1.0; // no overlap
            else {
                double ra = (1.0 / wa) / wa;
                double rb = (1.0 / wb) / wb;

                // lines have to cross
                triangular_distribution<> d1(std::nexttoward(a.avg() - wa, -std::numeric_limits<double>::infinity()), a.avg(),
                        std::nexttoward(a.avg() + wa, std::numeric_limits<double>::infinity()));
                triangular_distribution<> d2(std::nexttoward(b.avg() - wb, -std::numeric_limits<double>::infinity()), b.avg(),
                        std::nexttoward(b.avg() + wb, std::numeric_limits<double>::infinity()));


                // lets try all the extremes first
                for (auto x :{d1.lower(), d1.upper(), d2.lower(), d2.upper()}) {
                    dist = std::max(std::abs(cdf(d1, x) - cdf(d2, x)), dist);
                }

                if ((d1.lower() < d2.lower() && d2.lower() < d1.mode()) ||
                        (d2.lower() < d1.lower() && d1.lower() < d2.mode())) {
                    // left intersect left
                    auto inter = ((-d1.lower()) - (-d2.lower())) / (rb - ra);
                    if (inter <= std::max(d1.mode(), d2.mode()) && inter >= std::min(d1.mode(), d2.mode()))
                        dist = std::max(std::abs(cdf(d1, inter) - cdf(d2, inter)), dist);
                }

                if ((d1.upper() > d2.upper() && d2.upper() > d1.mode()) ||
                        (d2.upper() > d1.upper() && d1.upper() > d2.mode())) {
                    auto inter = ((-d1.upper()) - (-d2.upper())) / ((-rb)-(-ra));
                    if (inter <= std::max(d1.mode(), d2.mode()) && inter >= std::min(d1.mode(), d2.mode()))
                        dist = std::max(std::abs(cdf(d1, inter) - cdf(d2, inter)), dist);
                }

                if ((d1.lower() < d2.mode() && d1.mode() > d2.mode()) ||
                        (d2.upper() > d1.upper() && d2.mode() < d1.mode())) // d1 left intersects d2 right
                {
                    auto inter = ((-d1.lower()) - (-d2.upper())) / ((-rb) - ra);
                    if (inter <= std::max(d1.mode(), d2.mode()) && inter >= std::min(d1.mode(), d2.mode()))
                        dist = std::max(std::abs(cdf(d1, inter) - cdf(d2, inter)), dist);
                }

                if ((d1.upper() > d2.upper() && d1.mode() < d2.mode()) || // d1 right intersects d2 left
                        (d2.lower() < d1.mode() && d2.mode() > d1.mode())) {
                    auto inter = ((-d1.upper()) - (-d2.lower())) / (rb - (-ra));
                    if (inter <= std::max(d1.mode(), d2.mode()) && inter >= std::min(d1.mode(), d2.mode()))
                        dist = std::max(std::abs(cdf(d1, inter) - cdf(d2, inter)), dist);
                }
            }

            //        std::cerr << " DIST " << dist << " CA " << ca << " MES " << mes << " " << (ca*mes) << std::endl;
            if (dist > ca * mes) {
                _vfilter += (1.0 - _vfilter) * rate;
                _hfilter += (0.0 - _hfilter) * rate;
                _lfilter += (0.0 - _lfilter) * rate;
                //std::cerr << "SPLITF2 " << _vfilter << " " << _lfilter << " " << _hfilter << std::endl;
            }
        }
    }
}

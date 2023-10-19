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
 * File:   structs.h
 * Author: Peter G. Jensen
 *
 * Created on July 25, 2017, 10:31 AM
 */

#ifndef STRUCTS_H
#define STRUCTS_H


#include <memory>
#include <stddef.h>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <ostream>
#include <iostream>

namespace prlearn {

    struct avg_t {
        double _avg = 0;
        double _cnt = 0;

        constexpr avg_t() = default;
        constexpr avg_t(const avg_t&) = default;

        inline void addPoints(const avg_t& other) {
            addPoints(other._cnt, other._avg);
        }

        inline void addPoints(double weight, double d) {
            if (weight == 0) return;
            if (_cnt == 0) {
                _cnt = weight;
                _avg = d;
            } else {
                _cnt += weight;
                double diff = d - _avg;
                _avg += diff * (weight / _cnt); // add only "share" of difference
            }
            assert(!std::isnan(_avg));
        }

        inline void addPoint(double d) {
            addPoints(1, d);
        }

        inline avg_t& operator=(const avg_t& other) {
            _avg = other._avg;
            _cnt = other._cnt;
            assert(!std::isnan(_avg));
            return *this;
        }

        inline void reset() {
            _cnt = 0;
            _avg = 0;
        }

        avg_t& operator+=(const avg_t& other) {
            addPoints(other);
            return *this;
        }

        avg_t& operator+=(double d) {
            addPoints(1, d);
            return *this;
        }

        bool operator!=(const avg_t& other) const {
            return _cnt != other._cnt || _avg != other._avg;
        }
    };

    std::ostream& operator<<(std::ostream& stream, const avg_t& el);

    struct qvar_t : private avg_t {

        qvar_t() = default;

        qvar_t(double d, double w, double squared) {
            _avg = d;
            _cnt = w;
            _sq = squared;
        };
        // this is a dirty hijack!
        qvar_t& operator+=(double d);
        void addPoints(double weight, double d);

        auto& avg() {
            return _avg;
        }

        auto& cnt() {
            return _cnt;
        }

        auto& avg() const {
            return _avg;
        }

        auto& cnt() const {
            return _cnt;
        }

        bool operator!=(const qvar_t& other) const {
            return _cnt != other._cnt || _avg != other._avg;
        }
        void print(std::ostream& stream) const;
        static qvar_t approximate(const qvar_t& a, const qvar_t& b);
        double variance() const {
            auto pow = std::pow(_avg, 2.0);
            if(pow >= _sq)
                return 0;
            return _sq - pow;
        }

        void set_variance(double var) {
            _sq = std::pow(_avg, 2.0) + var;
        }

       double& squared() {
            return _sq;
       }

        const double& squared() const {
            return _sq;
        }

    private:
        double _sq = 0;
    };

    struct splitfilter_t {
        double _vfilter = 0.0;
        double _hfilter = 0.0;
        double _lfilter = 0.0;

        void reset() {
            _vfilter = _hfilter = _lfilter = 0;
        }

        double max() const {
            return std::max(_vfilter, std::max(_hfilter, _lfilter));
        }
        void add(const qvar_t&, const qvar_t&, double indif, double tl, double tu, double t2, double rate);
    };


    std::ostream& operator<<(std::ostream&, const qvar_t&);

    struct simple_split_t {
        size_t _var = 0;
        double _boundary = 0;
        size_t _low = 0;
        size_t _high = 0;
        bool _is_split = false;
    };

    struct el_t {
        el_t(const el_t&) = default;
        el_t(el_t&&) = default;
        el_t(size_t l);
        el_t& operator=(const el_t&) = default;
        el_t& operator=(el_t&&) = default;
        size_t _label;
        size_t _nid;
        bool operator<(const el_t& other) const;
    };
}
#endif /* STRUCTS_H */


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
 * File:   SimpleRegressor.h
 * Author: Peter G. Jensen
 *
 * Created on July 18, 2017, 1:23 PM
 */

#ifndef SIMPLEREGRESSOR_H
#define SIMPLEREGRESSOR_H

#include "propts.h"
#include "structs.h"

#include <limits>
#include <vector>
#include <map>
#include <iomanip>

namespace prlearn {

    class SimpleRegressor {
    public:
        SimpleRegressor() = default;
        SimpleRegressor(const SimpleRegressor&) = default;
        SimpleRegressor(SimpleRegressor&&) = default;

        qvar_t lookup(size_t label, const double*, size_t dimen) const {
            el_t lf(label);

            auto res = std::lower_bound(std::begin(_labels), std::end(_labels), lf);

            if (res != std::end(_labels) && res->_label == label)
                return qvar_t{res->_value.avg(), (double)res->_value.cnt(), res->_value._variance};
            else
                return qvar_t{std::numeric_limits<double>::quiet_NaN(), 0, 0};
        }

        double getBestQ(const double*, bool minimization, size_t* next_labels = nullptr, size_t n_labels = 0) const {
            double res = std::numeric_limits<double>::infinity();
            if (!minimization)
                res = -res;
            size_t j = 0;
            for (auto& e : _labels)
            {
                if(next_labels != nullptr)
                {
                    for(;j < n_labels && next_labels[j] < e._label; ++j) {}
                    if(j >= n_labels) return res;
                    if(next_labels[j] != e._label) continue;
                }
                if (!std::isinf(e._value.avg()) && !std::isnan(e._value.avg()))
                    res = minimization ?
                        std::min(res, e._value.avg()) :
                    std::max(res, e._value.avg());
            }
            return res;
        }

        void update(size_t label, const double*, size_t dimen, double nval, const double delta, const propts_t&) {
            el_t lf(label);

            auto res = std::lower_bound(std::begin(_labels), std::end(_labels), lf);

            if (res == std::end(_labels) || res->_label != label)
                res = _labels.insert(res, lf);
            res->_value += nval;
            assert(res->_value.avg() >= 0);
        }

        void print(std::ostream& s, size_t tabs, std::map<size_t, size_t>& label_map) const {
            s << std::setprecision (std::numeric_limits<double>::digits10 + 1);
            for (size_t i = 0; i < tabs; ++i) s << "\t";
            s << "{";
            bool first = true;
            for (auto& w : _labels) {
                if (!first) s << ",";
                first = false;
                s << "\n";
                for (size_t t = 0; t < tabs; ++t) s << "\t";
                s << "\"" << label_map[w._label] << "\" : ";
                auto v = w._value.avg();
                if(!std::isinf(v) && !std::isnan(v))
                    s << v;
                else
                    s << "\"inf\"";
            }
            s << "\n";
            for (size_t i = 0; i < tabs; ++i) s << "\t";
            s << "}";
        }

    protected:

        struct el_t {

            el_t(size_t label) : _label(label) {
            };

            el_t(size_t label, double d) : _label(label) {
                _value += d;
            };
            size_t _label = 0;
            rqvar_t _value;

            bool operator<(const el_t& other) const {
                return _label < other._label;
            }
        };
        std::vector<el_t> _labels;

    };

}
#endif /* SIMPLEREGRESSOR_H */


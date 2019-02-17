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

#include <limits>
#include <vector>
#include <map>

#include "avg.h"
#include "statmc/propts.h"


class SimpleRegressor
{
public:
    SimpleRegressor() = default;
    SimpleRegressor(const SimpleRegressor&) = default;
    SimpleRegressor(SimpleRegressor&&) = default;

    avg_t lookup(size_t label, const double*, size_t dimen) const
    {
        el_t lf(label);
        
        auto res = std::lower_bound(std::begin(_labels), std::end(_labels), lf);
        
        if(res != std::end(_labels) && res->_label == label) 
            return res->_value;
        else 
            return avg_t{std::numeric_limits<double>::quiet_NaN(), 0};
    }
    
    double getBestQ(const double*, bool minimization) const
    {
        double res = std::numeric_limits<double>::infinity();
        if(!minimization)
            res = -res;
        for(auto& e : _labels)
            if(!std::isinf(e._value._avg) && !std::isnan(e._value._avg))
                res = minimization ? 
                    std::min(res, e._value._avg) :
                    std::max(res, e._value._avg) ;
        return res;
    }
    
    void update(size_t label, const double*, size_t dimen, double nval, const double delta, const propts_t& options)
    {
        el_t lf(label);
        
        auto res = std::lower_bound(std::begin(_labels), std::end(_labels), lf);
        
        if(res == std::end(_labels) || res->_label != label) 
            res = _labels.insert(res, lf);
        res->_value._cnt = std::min<size_t>(res->_value._cnt, options._q_learn_rate);
        res->_value += nval;
        assert(res->_value._avg >= 0);
    }
        
    void print(std::ostream& s, size_t tabs, std::map<size_t,size_t>& label_map) const
    {
        for(size_t i = 0; i < tabs; ++i) s << "\t";
        s << "{";
        bool first = true;
        for(auto& w : _labels)
        {
            if(!first) s << ",";
            first = false;
            s << "\n";
            for(size_t t = 0; t < tabs; ++t) s << "\t";
            s << "\"" << label_map[w._label] << "\" : " << w._value._avg;
        }        
        s << "\n";
        for(size_t i = 0; i < tabs; ++i) s << "\t";
        s << "}";          
    }

protected:
    struct el_t { 
        el_t(size_t label) : _label(label) {};
        el_t(size_t label, double d) : _label(label) 
        {
            _value += d;
        };
        size_t _label;
        avg_t _value;
        bool operator<(const el_t& other) const
        {
            return _label < other._label;
        }
    };
    std::vector<el_t> _labels;

};


#endif /* SIMPLEREGRESSOR_H */


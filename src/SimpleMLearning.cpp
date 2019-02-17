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
 * File:   SimpleMLearning.cpp
 * Author: Peter G. Jensen
 *
 * Created on July 30, 2017, 10:02 AM
 */

#include "SimpleMLearning.h"

void SimpleMLearning::addSample(size_t, const double*, const double*, size_t label, size_t dest, double value, const std::vector<SimpleMLearning>& clouds, bool minimization, const double, const propts_t& options)
{
    node_t act;
    act._label = label;
    auto lb = std::lower_bound(std::begin(_nodes), std::end(_nodes), act);
    if(lb == std::end(_nodes) || lb->_label != act._label)
        lb = _nodes.insert(lb, act);
    
    succs_t succ;
    succ._nid = dest;
    auto le = std::lower_bound(std::begin(lb->_succssors), std::end(lb->_succssors), succ);
    if(le == std::end(lb->_succssors) || le->_nid != dest)
        le = lb->_succssors.insert(le, succ);
    value *= options._discount;
    le->_cost += value;
    
    update(clouds, minimization);

}

std::pair<double,double> SimpleMLearning::lookup(size_t label, const double*, size_t) const
{
    node_t lf;
    auto lb = std::lower_bound(std::begin(_nodes), std::end(_nodes), lf);
    if(lb == std::end(_nodes) || lb->_label != label)
        return std::make_pair(std::numeric_limits<double>::quiet_NaN(), 0.0);
    return std::make_pair(lb->_q._avg, lb->_q._cnt);
}

void SimpleMLearning::print(std::ostream& s, size_t tabs, std::map<size_t, size_t>& label_map, const std::vector<SimpleMLearning>&) const 
{
    for(size_t i = 0; i < tabs; ++i) s << "\t";
    s << "{";
    bool first = true;
    for(auto& el :  _nodes)
    {
        if(!first) s << ",";
        first = false;
        s << "\n";
        for(size_t i = 0; i < tabs+1; ++i) s << "\t";
        s << "\"";
        s << label_map[el._label];
        s << "\":" << el._q._avg;
    }
    s << "\n";
    for(size_t i = 0; i < tabs; ++i) s << "\t";
    s << "}";    
}

void SimpleMLearning::update(const std::vector<SimpleMLearning>& clouds, bool minimization) 
{
    avg_t rq;
    if(minimization) rq._avg = std::numeric_limits<double>::infinity();
    else             rq._avg = -std::numeric_limits<double>::infinity();
    for(auto& n : _nodes)
    {
        avg_t nq;

        for(auto& s : n._succssors)
        {
            auto dq = clouds[s._nid]._q._avg;
            if(std::isinf(dq) || std::isnan(dq)) dq = 0;
            nq.addPoints(s._cost._cnt, s._cost._avg + dq);
        }
        if( (minimization && nq._avg < rq._avg) ||
           (!minimization && nq._avg > rq._avg))
        {
            rq = nq;
        }
        n._q = nq;
    }
    _q = rq;
}

bool SimpleMLearning::succs_t::operator<(const succs_t& other) const {
    return _nid < other._nid;
}

bool SimpleMLearning::node_t::operator<(const node_t& other) const {
    return _label < other._label;
}






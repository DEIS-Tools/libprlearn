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
 * File:   RefinementTree.cpp
 * Author: Peter G. Jensen
 *
 * Created on July 18, 2017, 5:09 PM
 */


#include "RefinementTree.h"
#include <limits>
#include <iomanip>
#include <iostream>

namespace prlearn {

    RefinementTree::RefinementTree() {
    }

    void RefinementTree::print(std::ostream& s, size_t tabs, std::map<size_t, size_t>& edge_map) const {
        s << std::setprecision (std::numeric_limits<double>::digits10 + 1);
        for (size_t i = 0; i < tabs; ++i) s << "\t";
        s << "{";
        bool first = true;
        for (auto& el : _mapping) {
            if (!first) s << ",";
            first = false;
            s << "\n";
            for (size_t i = 0; i < tabs + 1; ++i) s << "\t";
            s << "\"";
            s << edge_map[el._label];
            s << "\":\n";
            _nodes[el._nid].print(s, tabs + 2, _nodes);
        }
        s << "\n";
        for (size_t i = 0; i < tabs; ++i) s << "\t";
        s << "}";
    }

    RefinementTree::RefinementTree(const RefinementTree& other) {
        _dimen = other._dimen;
        _nodes.reserve(other._nodes.size());
        _mapping = other._mapping;

        for (auto& n : other._nodes)
            _nodes.emplace_back(n, _dimen);
    }

    qvar_t
    RefinementTree::lookup(size_t label, const double* point, size_t) const {
        el_t lf(label);
        auto res = std::lower_bound(std::begin(_mapping), std::end(_mapping), lf);
        if (res == std::end(_mapping) || res->_label != label)
            return qvar_t(std::numeric_limits<double>::quiet_NaN(), 0, 0);
        auto n = _nodes[res->_nid].get_leaf(point, res->_nid, _nodes);
        auto& node = _nodes[n];
        return qvar_t(node._predictor._q.avg(), node._predictor._q.cnt(), node._predictor._q.squared());
    }

    double RefinementTree::getBestQ(const double* point, bool minimization, size_t* next_labels, size_t n_labels) const {
        auto val = std::numeric_limits<double>::infinity();
        if (!minimization)
            val = -val;
        if(next_labels == nullptr)
        {
            for (const el_t& el : _mapping) {
                auto node = _nodes[el._nid].get_leaf(point, el._nid, _nodes);
                auto v = _nodes[node]._predictor._q.avg();
                if (!std::isinf(v) && !std::isnan(v))
                    val = minimization ?
                        std::min(v, val) :
                    std::max(v, val);
            }
        }
        else {
            size_t j = 0;
            for(size_t i = 0; i < n_labels; ++i)
            {
                for(;j < _mapping.size() && _mapping[j]._label < next_labels[i]; ++j) {};
                if(j >= _mapping.size()) return val;
                if(_mapping[j]._label != next_labels[i])
                    continue;
                const auto& res = _mapping[j];
                auto node = _nodes[res._nid].get_leaf(point, res._nid, _nodes);
                auto v = _nodes[node]._predictor._q.avg();
                if (!std::isinf(v) && !std::isnan(v))
                    val = minimization ?
                        std::min(v, val) :
                    std::max(v, val);
            }
        }
        return val;
    }

    void
    RefinementTree::update(size_t label, const double* point, size_t dimen, double nval, const double delta, const propts_t& options) {
        _dimen = dimen;
        el_t lf(label);
        auto res = std::lower_bound(std::begin(_mapping), std::end(_mapping), lf);
        if (res == std::end(_mapping) || res->_label != label) {
            lf._nid = _nodes.size();
            _nodes.emplace_back();
            res = _mapping.insert(res, lf);
        }

        assert(res->_label == label);
        auto n = _nodes[res->_nid].get_leaf(point, res->_nid, _nodes);
        _nodes[n].update(point, dimen, nval, _nodes, delta, options);
    }

    RefinementTree::node_t::node_t(const node_t& other, size_t dimen) {
        _predictor = qpred_t(other._predictor, dimen);
        _split = other._split;
    }

    void RefinementTree::node_t::print(std::ostream& s, size_t tabs, const std::vector<node_t>& nodes) const {
        for (size_t i = 0; i < tabs; ++i) s << "\t";
        if (_split._is_split) {
            s << "{\"var\":" << _split._var << ",\"bound\":" << _split._boundary << ",\n";
            for (size_t i = 0; i < tabs + 1; ++i) s << "\t";
            s << "\"low\":\n";
            nodes[_split._low].print(s, tabs + 2, nodes);
            s << ",\n";
            for (size_t i = 0; i < tabs + 1; ++i) s << "\t";
            s << "\"high\":\n";
            nodes[_split._high].print(s, tabs + 2, nodes);
            s << "\n";
            for (size_t i = 0; i < tabs; ++i) s << "\t";
            s << "}";
        } else {
            auto v = _predictor._q.avg();
            if(!std::isinf(v) && !std::isnan(v))
                s << _predictor._q.avg();
            else
                s << "\"inf\"";
        }
    }

    size_t RefinementTree::node_t::get_leaf(const double* point, size_t current, const std::vector<node_t>& nodes) const {
        if (!_split._is_split) return current;
        if (point[_split._var] <= _split._boundary)
            return nodes[_split._low].get_leaf(point, _split._low, nodes);
        else
            return nodes[_split._high].get_leaf(point, _split._high, nodes);
    }

    void RefinementTree::node_t::update(const double* point, size_t dimen, double nval, std::vector<node_t>& nodes, double delta, const propts_t& options) {
        assert(!_split._is_split);
        if (_predictor._data == nullptr)
            _predictor._data = std::make_unique < qdata_t[]>(dimen);
        _predictor._q += nval;
        auto svar = 0;
        auto cnt = 0;

        for (size_t i = 0; i < dimen; ++i) {
            // add new data-point to all hypothetical new partitions
            if (point[i] <= _predictor._data[i]._midpoint._avg) {
                _predictor._data[i]._lowq += nval;
                _predictor._data[i]._lmid += point[i];
            } else {
                _predictor._data[i]._highq += nval;
                _predictor._data[i]._hmid += point[i];
            }

            // update the split-filters
            _predictor._data[i]._splitfilter.add(_predictor._data[i]._lowq,
                    _predictor._data[i]._highq,
                    delta * options._indefference,
                    options._lower_t,
                    options._upper_t,
                    options._ks_limit,
                    options._filter_rate);

            // if the critical value is reached by any of the three split-conditions,
            // we split. Notice the random choice - we want to avoid bias.
            if (_predictor._data[i]._splitfilter.max() >= options._filter_val) {
                ++cnt;
                if ((std::rand() % cnt) == 0)
                    svar = i;
            }
        }

        // only true if some candidate exceeded the critical value.
        if (cnt > 0) {
            _split._is_split = true;
            _split._var = svar;
            _split._boundary = _predictor._data[svar]._midpoint._avg;
            auto slow = _split._low = nodes.size();
            auto shigh = _split._high = nodes.size() + 1;
            std::unique_ptr < qdata_t[] > tmp;
            tmp.swap(_predictor._data);
            auto oq = _predictor._q;

            // this  <-- is invalidated below!
            nodes.emplace_back();
            nodes.emplace_back();
            nodes[slow]._predictor._q = tmp[svar]._lowq;
            nodes[shigh]._predictor._q = tmp[svar]._highq;
            nodes[slow]._predictor._data = std::make_unique < qdata_t[]>(dimen);
            nodes[shigh]._predictor._data = std::make_unique < qdata_t[]>(dimen);
            for (int i = 0; i < (int) dimen; ++i) {
                if (i == svar) {
                    nodes[slow]._predictor._data[i]._midpoint = tmp[i]._lmid;
                    nodes[shigh]._predictor._data[i]._midpoint = tmp[i]._hmid;
                } else {
                    auto tmid = tmp[i]._lmid;
                    tmid += tmp[i]._hmid;
                    nodes[slow]._predictor._data[i]._midpoint = tmid;
                    nodes[shigh]._predictor._data[i]._midpoint = tmid;
                }
            }
            if (oq.cnt() > 0) {
                if (nodes[slow]._predictor._q.cnt() == 0) {
                    nodes[slow]._predictor._q.cnt() = 1;
                    nodes[slow]._predictor._q.avg() = oq.avg();
                    nodes[slow]._predictor._q.squared() = std::pow(oq.avg(), 2.0);
                }
                if (nodes[shigh]._predictor._q.cnt() == 0) {
                    nodes[shigh]._predictor._q.cnt() = 1;
                    nodes[shigh]._predictor._q.avg() = oq.avg();
                    nodes[shigh]._predictor._q.squared() = std::pow(oq.avg(), 2.0);
                }
            }
            assert(nodes[shigh]._predictor._q.cnt() > 0);
            assert(nodes[slow]._predictor._q.cnt() > 0);
        } else {
            // does not improve learning.
            // check split-bounds, reset if needed
            if (_predictor._data) {
                bool rezero = false;
                for (size_t i = 0; i < dimen; ++i) {
                    auto& dp = _predictor._data[i];
                    auto mx = std::max(dp._hmid._cnt, dp._lmid._cnt);
                    auto mn = std::min(dp._hmid._cnt, dp._lmid._cnt);
                    if (mx >= 2 && std::pow(5, mn) < mx && mx > dp._midpoint._cnt) {
                        // update split-bound
                        auto nm = dp._lmid;
                        nm += dp._hmid;
                        if (nm._avg == dp._midpoint._avg)
                            continue;

                        rezero = true;
                        dp._hmid = nm;
                        dp._lmid = nm;
                        dp._hmid._cnt /= 2;
                        dp._lmid._cnt /= 2;
                        dp._midpoint += nm;

                        // merge Q-values. TODO: See if this cannot be done better.
                        dp._lowq = qvar_t::approximate(dp._lowq, dp._highq);
                        dp._lowq.cnt() /= 2;
                        dp._highq = dp._lowq;
                    }
                }
                // If any was reset, reset all split-counters.
                // We have to reset all to avoid introducing bias.
                if (rezero) {
                    for (size_t i = 0; i < dimen; ++i)
                        _predictor._data[i]._splitfilter.reset();
                }
            }
        }
    }
}

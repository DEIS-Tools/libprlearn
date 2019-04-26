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
 * File:   MLearning.cpp
 * Author: Peter G. Jensen
 * 
 * Created on July 25, 2017, 9:58 AM
 */


#include "MLearning.h"

#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>

namespace prlearn {

    bool MLearning::interesect_t::operator<(const interesect_t& other) const {
        if (_size != other._size) return _size < other._size;
        if (_cloud != other._cloud) return _cloud < other._cloud;
        for (size_t i = 0; i < _size; ++i) {
            if (_nodes[i] != other._nodes[i])
                return _nodes[i] < other._nodes[i];
        }
        return false;
    }

    bool MLearning::interesect_t::operator!=(const interesect_t& other) const {
        if (_size != other._size) return true;
        if (_cloud != other._cloud) return true;
        for (size_t i = 0; i < _size; ++i) {
            if (_nodes[i] != other._nodes[i]) return true;
        }
        return false;
    }

    MLearning::interesect_t::interesect_t(const interesect_t& other, size_t dimen) {
        _size = other._size;
        _cloud = other._cloud;
        if (other._nodes != nullptr) {
            _nodes = std::make_unique < size_t[]>(_size);
            memcpy(_nodes.get(), other._nodes.get(), _size * sizeof (size_t));
        }

        assert(_size == 0 || this->_nodes != nullptr);
        assert(_cloud != 0 || _size == 0);

        if (other._variance != nullptr) {
            _variance = std::make_unique < std::pair<qvar_t, qvar_t>[]>(dimen);
            for (size_t i = 0; i < dimen; ++i) {
                _variance[i] = other._variance[i];
                assert(_variance[i].first.avg() == other._variance[i].first.avg());
                assert(_variance[i].second.avg() == other._variance[i].second.avg());
            }
        }
        if (other._old != nullptr) {
            _old = std::make_unique < std::pair<qvar_t, qvar_t>[]>(dimen);
            for (size_t i = 0; i < dimen; ++i)
                _old[i] = other._old[i];
        }
    }

    MLearning::node_t::node_t(const node_t& other, size_t dimen) {
        _split = other._split;
        _q = other._q;
        _old = other._old;
        _samples.reserve(other._samples.size());
        _parent = other._parent;
        for (auto& s : other._samples)
            _samples.emplace_back(s, dimen);
        if (other._data) {
            _data = std::make_unique < data_t[]>(dimen);
            for (size_t i = 0; i < dimen; ++i)
                _data[i] = other._data[i];
        }
    }

    MLearning::MLearning(const MLearning& other) {
        _dimen = other._dimen;
        _mapping = other._mapping;
        for (auto& n : other._nodes)
            _nodes.emplace_back(n, _dimen);
    }

    MLearning::MLearning() {
    }

    void MLearning::addSample(size_t dimen, const double* f_var,
            const double* t_var, size_t label,
            size_t dest, double value, const std::vector<MLearning>& clouds,
            bool minimization, const double delta,
            const propts_t& options) {
        _dimen = dimen;
        el_t lf((size_t) label);
        auto res = _mapping.end();
        for (size_t i = 0; i < _mapping.size(); ++i) {
            if (lf._label == _mapping[i]._label) {
                res = _mapping.begin() + i;
                break;
            }
        }
        if (res == _mapping.end()) {
            lf._nid = _nodes.size();
            _nodes.emplace_back();
            res = _mapping.insert(res, lf);
            _nodes[res->_nid]._parent = lf._nid; // self loop in root
        }

        assert(res->_label == (size_t) label);

        auto node = _nodes[res->_nid].find_node(_nodes, f_var, res->_nid);
        assert(node < _nodes.size());
        _nodes[node].add_sample(dest, f_var, t_var, value, _dimen, clouds);
        _nodes[node].update(node, minimization, clouds, _nodes, dimen, true, delta, options);

        if (_mapping.size() <= 1) return;
        auto bv = std::numeric_limits<double>::infinity();
        if (!minimization)
            bv = -bv;

        std::vector<size_t> best;
        size_t rnd = 0;
        size_t fcnt = 0;
        for (size_t i = 0; i < _mapping.size(); ++i) {
            if (res->_nid == _mapping[i]._nid) continue;
            auto nn = _nodes[_mapping[i]._nid].find_node(_nodes, f_var, _mapping[i]._nid);
            if ((minimization && bv >= _nodes[nn]._q.avg()) ||
                    (!minimization && bv <= _nodes[nn]._q.avg())) {
                if (bv != _nodes[nn]._q.avg()) {
                    for (auto r : best) {
                        ++fcnt;
                        if ((std::rand() % fcnt) == 0)
                            rnd = r;
                    }
                    best.clear();
                }
                best.push_back(nn);
                bv = _nodes[nn]._q.avg();
            } else {
                ++fcnt;
                if ((std::rand() % fcnt) == 0)
                    rnd = nn;
            }
        }
        for (auto best_alt : best)
            _nodes[best_alt].update(best_alt, minimization, clouds, _nodes, dimen, false, delta, options);
        if (fcnt > 0)
            _nodes[rnd].update(rnd, minimization, clouds, _nodes, dimen, false, delta, options);
    }

    qvar_t MLearning::lookup(size_t label, const double* f_var, size_t dimen) const {
        for (auto& el : _mapping) {
            if (el._label == label) {
                auto n = _nodes[el._nid].find_node(_nodes, f_var, el._nid);
                return _nodes[n]._q;
            }
        }
        return qvar_t(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0);
    }

    void MLearning::print(std::ostream& s, size_t tabs, std::map<size_t, size_t>& edge_map, const std::vector<MLearning>& clouds) const {
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

    void MLearning::node_t::print(std::ostream& s, size_t tabs, const std::vector<node_t>& nodes) const {
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
            for (size_t i = 0; i < tabs; ++i) s << "\t";
            s << _q.avg();
        }
    }

    void MLearning::update(const std::vector<MLearning>& clouds, bool minimization) 
    {
        std::cerr << "SIZE " << this << " " << _nodes.size() << std::endl;
    }

    std::unique_ptr<size_t[] > MLearning::findIntersection(const double* point) const {
        auto target = std::make_unique < size_t[]>(_mapping.size());
        for (size_t i = 0; i < _mapping.size(); ++i) {
            target[i] = _nodes[_mapping[i]._nid].find_node(_nodes, point, _mapping[i]._nid);
        }
        return target;
    }

    std::pair<qvar_t, qvar_t> MLearning::node_t::aggregate_samples(const std::vector<MLearning>& clouds, size_t dimen, bool minimize, std::pair<qvar_t, qvar_t>* tmpq, double discount) {
        avg_t mean, old_mean;
        std::vector<qvar_t> sample_qvar;
        std::vector<qvar_t> old_var;
        double fut = 0;
        for (auto& s : _samples) {
            auto best = minimize ? std::numeric_limits<double>::infinity() :
                    -std::numeric_limits<double>::infinity();
            double var = 0;
            if (s._size == 0 || s._cloud == 0 || discount == 0) {
                best = 0;
            } else {
                for (size_t i = 0; i < s._size; ++i) {
                    assert(s._nodes[i] < clouds[s._cloud]._nodes.size());
                    auto c = clouds[s._cloud]._nodes[s._nodes[i]]._q.avg();
                    fut = std::min(fut, c);
                    if (c == best)
                        var = std::min(var, clouds[s._cloud]._nodes[s._nodes[i]]._q._variance);
                    else if ((c < best && minimize) || (c > best && !minimize)) {
                        best = c;
                        var = clouds[s._cloud]._nodes[s._nodes[i]]._q._variance;
                    }
                }
            }

            assert(!std::isinf(best));
            assert(!std::isnan(best));
            // avoid negative (/positive) cycles when minimizing/maximizing.
            best *= discount;
            // dont look too far into the future for the variance.
            // if we do, it will grow in horrible ways and be useless.
            var *= std::min(0.5, discount);
            for (size_t d = 0; d < dimen; ++d) {
                if (s._variance) {
                    auto v = s._variance[d];
                    v.first.avg() += best;
                    v.second.avg() += best;
                    v.first._variance = std::max(v.first._variance, var);
                    v.second._variance = std::max(v.second._variance, var);
                    tmpq[d].first.addPoints(v.first.cnt(), v.first.avg());
                    tmpq[d].second.addPoints(v.second.cnt(), v.second.avg());
                    mean.addPoints(v.first.cnt(), v.first.avg());
                    mean.addPoints(v.second.cnt(), v.second.avg());
                    sample_qvar.emplace_back(v.first);
                    sample_qvar.emplace_back(v.second);
                }
                if (s._old) {
                    auto v = s._old[d];
                    v.first.avg() += best;
                    v.second.avg() += best;
                    v.first._variance = std::max(v.first._variance, var);
                    v.second._variance = std::max(v.second._variance, var);
                    old_mean.addPoints(v.first.cnt(), v.first.avg());
                    old_mean.addPoints(v.second.cnt(), v.second.avg());
                    old_var.push_back(v.first);
                    old_var.push_back(v.second);
                }
            }
        }

        avg_t svar, ovar;
        auto vars = std::make_unique < avg_t[]>(dimen * 2);
        bool first = true;
        size_t dimcnt = 0;
        for (auto& s : sample_qvar) {
            {
                const auto dif = std::abs(s.avg() - mean._avg);
                const auto std = std::sqrt(s._variance);
                auto var = (std::pow(dif + std, 2.0) + std::pow(dif - std, 2.0)) / 2.0;
                svar.addPoints(s.cnt(), var);
            }
            auto id = dimcnt;
            auto dmin = tmpq[id].first.avg();
            if (!first) {
                dmin = tmpq[dimcnt].second.avg();
                id = dimen + dimcnt;
            }
            {
                const auto dif = std::abs(s.avg() - dmin);
                const auto std = std::sqrt(s._variance);
                auto var = (std::pow(dif + std, 2.0) + std::pow(dif - std, 2.0)) / 2.0;
                vars[id].addPoints(s.cnt(), var);
            }
            if (!first)
                dimcnt = (dimcnt + 1) % dimen;
            first = !first;
        }

        for (auto& s : old_var) {
            const auto dif = std::abs(s.avg() - old_mean._avg);
            const auto std = std::sqrt(s._variance);
            auto var = (std::pow(dif + std, 2.0) + std::pow(dif - std, 2.0)) / 2.0;
            ovar.addPoints(s.cnt(), var);
        }

        for (size_t i = 0; i < dimen; ++i) {
            tmpq[i].first._variance = vars[i]._avg;
            tmpq[i].second._variance = vars[i + dimen]._avg;
        }

        qvar_t nq(mean._avg, mean._cnt / (dimen * 2), svar._avg);
        qvar_t oq(old_mean._avg, old_mean._cnt / (dimen * 2), ovar._avg);
        return std::make_pair(nq, oq);
    }

    void MLearning::node_t::update_parents(std::vector<node_t>& nodes, size_t next, bool minimize) {
        if (!nodes[next]._split._is_split)
            return;

        if ((nodes[nodes[next]._split._low]._q.avg() > nodes[nodes[next]._split._high]._q.avg()) == minimize)
            nodes[next]._q = nodes[nodes[next]._split._low]._q;
        else
            nodes[next]._q = nodes[nodes[next]._split._high]._q;
        assert(next < nodes.size());
        if (next == nodes[next]._parent)
            return;
        update_parents(nodes, nodes[next]._parent, minimize);
    }

    void MLearning::node_t::tighten_samples(const std::vector<MLearning>& clouds, size_t) {
        size_t i = 0;
        while (i < _samples.size()) {
            auto pointsize = clouds[_samples[i]._cloud]._mapping.size();

            assert(_samples[i]._size <= pointsize);
            if (pointsize == _samples[i]._size) {
                ++i;
                continue;
            }
            interesect_t tmp;
            tmp._size = pointsize;
            tmp._nodes = std::make_unique < size_t[]>(pointsize);
            tmp._cloud = _samples[i]._cloud;
            tmp._variance.swap(_samples[i]._variance);
            tmp._old.swap(_samples[i]._old);
            memcpy(tmp._nodes.get(), _samples[i]._nodes.get(), _samples[i]._size * sizeof (size_t));
            for (size_t j = _samples[i]._size; j < pointsize; ++j) {
                // TODO, improve, we know it has to be the smallest super-set node of the other nodes.
                auto& el = clouds[_samples[i]._cloud]._mapping[j];
                tmp._nodes[j] = el._nid;
            }

            _samples.erase(_samples.begin() + i);
            auto lb = std::lower_bound(_samples.begin(), _samples.end(), tmp);
            lb = _samples.emplace(lb, std::move(tmp));
            assert(lb->_size == pointsize);
        }
    }

    void MLearning::node_t::add_sample(size_t dest, const double* f_var, const double* t_var, double value, size_t dimen, const std::vector<MLearning>& clouds) {
        tighten_samples(clouds, dest);
        auto lb = _samples.begin();
        {
            interesect_t tmp;
            tmp._nodes = clouds[dest].findIntersection(t_var);
            tmp._cloud = dest;
            tmp._size = clouds[dest]._mapping.size();
            lb = std::lower_bound(_samples.begin(), _samples.end(), tmp);
            if (lb == _samples.end() || *lb != tmp)
                lb = _samples.emplace(lb, std::move(tmp));
        }

        if (lb->_variance == nullptr) {
            lb->_variance = std::make_unique < std::pair<qvar_t, qvar_t>[]>(dimen);
            for (size_t i = 0; i < dimen; ++i) {
                assert(lb->_variance[i].first.avg() == 0);
                assert(lb->_variance[i].first.cnt() == 0);
                assert(lb->_variance[i].second.avg() == 0);
                assert(lb->_variance[i].second.cnt() == 0);
            }
        }
        if (_data == nullptr)
            _data = std::make_unique < data_t[]>(dimen);

        for (size_t d = 0; d < dimen; ++d) {
            if (f_var[d] <= _data[d]._mid._avg) {
                lb->_variance[d].first += value;
                _data[d]._lmid += f_var[d];
            } else {
                lb->_variance[d].second += value;
                _data[d]._hmid += f_var[d];
            }
        }
    }

    void MLearning::node_t::update(size_t id, bool minimize, const std::vector<MLearning>& clouds, std::vector<node_t>& nodes, size_t dimen, bool allowSplit, const double delta, const propts_t& options) {
        assert(std::is_sorted(_samples.begin(), _samples.end()));
        assert(id < nodes.size());
        // Bellman update, compute "optimal" futures
        {
            auto tmpq = std::make_unique < std::pair<qvar_t, qvar_t>[]>(dimen);
            auto tmp = aggregate_samples(clouds, dimen, minimize, tmpq.get(), options._discount);
            tmp.second.cnt() = tmp.second.cnt() / 2.0;
            if (tmp.second.cnt() > tmp.first.cnt()) {
                tmp.second.cnt() -= tmp.first.cnt();
                tmp.first = qvar_t::approximate(tmp.second, tmp.first);
            } else if (tmp.second.cnt() > 0 && tmp.second.cnt() <= tmp.first.cnt()) {
                // clear out old
                for (int i = _samples.size() - 1; i >= 0; --i) {
                    _samples[i]._old = nullptr;
                    if (_samples[i]._variance == nullptr)
                        _samples.erase(_samples.begin() + i);
                }
            }
            _q = tmp.first;

            // cache Q and compute split!
            size_t svar = std::numeric_limits<size_t>::max();
            size_t cnt = 0;
            if (allowSplit) {
                if (_data == nullptr)
                    _data = std::make_unique < data_t[]>(dimen);
                for (size_t i = 0; i < dimen; ++i) {
                    _data[i]._splitfilter.add(tmpq[i].first,
                            tmpq[i].second,
                            delta * options._indefference,
                            options._lower_t,
                            options._upper_t,
                            options._ks_limit,
                            options._filter_rate);
                    if (_data[i]._splitfilter.max() >= options._filter_val) {
                        ++cnt;
                        if ((std::rand() % cnt) == 0)
                            svar = i;
                    }
                }
            }

            assert(!std::isnan(_q.avg()));
            if (cnt == 0 || allowSplit)
                update_parents(nodes, _parent, minimize);
            if (cnt == 0 && allowSplit) {
                // see if we need some readjustments here.
                for (size_t i = 0; i < dimen; ++i) {
                    auto& dp = _data[i];
                    auto mx = std::max(dp._hmid._cnt, dp._lmid._cnt);
                    auto mn = std::min(dp._hmid._cnt, dp._lmid._cnt);
                    if (mx >= 10 && std::pow(5, mn) < mx && mx > dp._mid._cnt * 2) {
                        // update split-bound
                        auto tmp = _data[i]._lmid;
                        tmp += _data[i]._hmid;
                        if (tmp._avg != _data[i]._mid._avg) {
                            tmp += _data[i]._mid;
                            _data[i] = data_t(); // clear old, set new mid, continue
                            _data[i]._mid = tmp;
                            for (auto& s : _samples) {
                                if (s._variance) {
                                    s._variance[i].first.cnt() = 0;
                                    s._variance[i].second.cnt() = 0;
                                    bool ok = false;
                                    for (size_t j = 0; j < dimen; ++j) {
                                        if (s._variance[j].first.cnt() != 0 ||
                                                s._variance[j].second.cnt() != 0)
                                            ok = true;
                                    }
                                    if (!ok)
                                        s._variance = nullptr;
                                }
                                if (s._old) {
                                    s._old[i].first.cnt() = 0;
                                    s._old[i].second.cnt() = 0;
                                    bool ok = false;
                                    for (size_t j = 0; j < dimen; ++j) {
                                        if (s._old[j].first.cnt() != 0 ||
                                                s._old[j].second.cnt() != 0)
                                            ok = true;
                                    }
                                    if (!ok)
                                        s._old = nullptr;
                                }
                            }
                        }
                    }
                }
                for (int i = _samples.size() - 1; i >= 0; --i) {
                    if (_samples[i]._variance == nullptr && _samples[i]._old == nullptr)
                        _samples.erase(_samples.begin() + i);
                }
            }
            else if (cnt > 0) {
                // SPLIT!
                _split._is_split = true;
                _split._var = svar; //sv.first;
                _split._boundary = _data[svar]._mid._avg;
                assert(!std::isnan(_split._boundary));
                auto slow = _split._low = nodes.size();
                auto shigh = _split._high = nodes.size() + 1;
                std::vector<interesect_t> samples;
                _samples.swap(samples);
                std::unique_ptr < data_t[] > data;
                data.swap(_data);
                // this  <-- is invalidated below invalid!
                nodes.emplace_back();
                nodes.emplace_back();

                nodes[slow]._q = tmpq[svar].first;
                nodes[shigh]._q = tmpq[svar].second;
                nodes[slow]._old = tmpq[svar].first;
                nodes[shigh]._old = tmpq[svar].second;
                nodes[slow]._parent = id;
                nodes[shigh]._parent = id;
                nodes[slow]._data = std::make_unique < data_t[]>(dimen);
                nodes[shigh]._data = std::make_unique < data_t[]>(dimen);
                for (size_t i = 0; i < dimen; ++i) {
                    if (i == svar) {
                        nodes[slow]._data[i]._mid = data[i]._lmid;
                        nodes[shigh]._data[i]._mid = data[i]._hmid;
                    } else {
                        auto tmid = data[i]._lmid;
                        tmid += data[i]._hmid;
                        tmid._cnt /= 2;
                        nodes[slow]._data[i]._mid = data[i]._lmid;
                        nodes[shigh]._data[i]._mid = data[i]._hmid;
                    }
                }

                // copy over samples here!
                for (auto& s : samples) {

                    if (s._variance != nullptr) {
                        double frac = s._variance[svar].second.cnt();
                        if (s._variance[svar].first.cnt() + s._variance[svar].second.cnt() == 0)
                            continue;
                        frac /= (double) (s._variance[svar].first.cnt() + s._variance[svar].second.cnt());
                        assert(frac <= 1);
                        for (auto n :{slow, shigh}) {
                            frac = 1.0 - frac;
                            if (n == slow && s._variance[svar].first.cnt() == 0)
                                continue;
                            if (n == shigh && s._variance[svar].second.cnt() == 0)
                                continue;
                            nodes[n]._samples.emplace_back(s, dimen);
                            auto& ns = nodes[n]._samples.back();
                            auto& nsv = ns._variance;
                            if (n == slow)
                                nsv[svar].second = nsv[svar].first;
                            else
                                nsv[svar].first = nsv[svar].second;
                            nsv[svar].first.cnt() = nsv[svar].second.cnt() = nsv[svar].first.cnt() / 2.0;
                            for (size_t i = 0; i < dimen; ++i) {
                                if (i == svar) continue;
                                nsv[i].first.cnt() = frac * nsv[i].first.cnt();
                                nsv[i].second.cnt() = frac * nsv[i].second.cnt();
                            }
                            ns._old.swap(ns._variance);
                            ns._variance = nullptr;
                            assert(std::is_sorted(nodes[n]._samples.begin(), nodes[n]._samples.end()));
                        }
                    }
                }
                nodes[id].update_parents(nodes, id, minimize);
            }
        }
    }

    size_t MLearning::node_t::find_node(const std::vector<node_t>& nodes, const double* point, const size_t id) const {
        if (_split._is_split) {
            auto next = point[_split._var] <= _split._boundary ? _split._low : _split._high;
            return nodes[next].find_node(nodes, point, next);
        } else {
            return id;
        }
    }

}
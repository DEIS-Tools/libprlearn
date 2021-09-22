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
 * File:   RefinementTree.h
 * Author: Peter G. Jensen
 *
 * Created on July 18, 2017, 5:09 PM
 */

#ifndef REFINEMENTTREE_H
#define REFINEMENTTREE_H

#include <vector>
#include <limits>
#include <map>

#include "structs.h"
#include "propts.h"

namespace prlearn {

    class RefinementTree {
    public:
        RefinementTree();

        RefinementTree(const RefinementTree&);
        RefinementTree(RefinementTree&&) = default;

        qvar_t lookup(size_t label, const double*, size_t dimen) const;

        void update(size_t label, const double*, size_t dimen, double nval, const double delta, const propts_t& options);

        void print(std::ostream& s, size_t tabs, std::map<size_t, size_t>& edge_map) const;

        double getBestQ(const double* val, bool minimization) const;

    protected:

        struct qdata_t {
            avg_t _midpoint;
            avg_t _lmid, _hmid;
            qvar_t _lowq, _highq;
            splitfilter_t _splitfilter;
        };

        struct qpred_t {
            qpred_t() = default;
            qpred_t(qpred_t&&) = default;
            qpred_t& operator=(qpred_t&&) = default;

            qpred_t(const qpred_t& other, size_t dimen) {
                _q = other._q;
                if (other._data) {
                    _data = std::make_unique < qdata_t[]>(dimen);
                    for (size_t i = 0; i < dimen; ++i)
                        _data[i] = other._data[i];
                }
            }
            qvar_t _q;
            size_t _cnt = 0;
            std::unique_ptr<qdata_t[] > _data = nullptr;
        };

        struct node_t {
            // we could do these two values as a single pointer 
            // which dynamically allocates enough space for both split and pred_t
            // including space for the run-time sized arrays.
            // however, this is at current time of writing a premature optimization.
            simple_split_t _split;
            qpred_t _predictor;

            size_t get_leaf(const double* point, size_t current, const std::vector<node_t>& nodes) const;
            void update(const double* point, size_t dimen, double nval, std::vector<node_t>& nodes, double delta, const propts_t& options);
            void print(std::ostream& s, size_t tabs, const std::vector<node_t>& nodes) const;

            node_t() = default;
            node_t(const node_t& other, size_t dimen);
            node_t(node_t&& other) noexcept = default;
            node_t& operator=(node_t&& other) = default;
        };

        std::vector<el_t> _mapping;
        std::vector<node_t> _nodes;
        size_t _dimen = 0;
    };

}

#endif /* REFINEMENTTREE_H */


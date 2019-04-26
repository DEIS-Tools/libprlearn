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
 * File:   MLearning.h
 * Author: Peter G. Jensen
 *
 * Created on July 25, 2017, 9:58 AM
 */

#ifndef MLEARNING_H
#define MLEARNING_H

#include <map>

#include "propts.h"
#include "structs.h"

namespace prlearn {

    class MLearning {
    public:
        MLearning();
        MLearning(const MLearning& other);
        MLearning& operator=(const MLearning& other) = default;

        MLearning(MLearning&&) = default;
        MLearning& operator=(MLearning&&) = default;

        void addSample(size_t dimen, // dimensions
                const double* f_var, const double* t_var, // doubles
                size_t label, // edge chosen, edge taken
                size_t dest, double value, // cost
                const std::vector<MLearning>& clouds, // other points
                bool minimization, const double delta,
                const propts_t& options
                );

        void update(const std::vector<MLearning>& clouds, bool minimization);

        qvar_t lookup(size_t label, const double* f_var, size_t dimen) const;

        void print(std::ostream& s, size_t tabs, std::map<size_t, size_t>& edge_map, const std::vector<MLearning>& clouds) const;

    protected:

        std::unique_ptr<size_t[] > findIntersection(const double* point) const;

        struct interesect_t {
            size_t _size = 0;
            size_t _cloud = std::numeric_limits<size_t>::max();
            std::unique_ptr<size_t[] > _nodes = nullptr;
            std::unique_ptr<std::pair<qvar_t, qvar_t>[] > _variance = nullptr;
            std::unique_ptr<std::pair<qvar_t, qvar_t>[] > _old = nullptr;

            interesect_t() = default;
            interesect_t(interesect_t&&) = default;
            interesect_t& operator=(interesect_t&&) = default;
            interesect_t(const interesect_t& other, size_t dimen);
            bool operator<(const interesect_t& other) const;
            bool operator!=(const interesect_t& other) const;
        };

        struct data_t {
            avg_t _lmid, _hmid, _mid;
            splitfilter_t _splitfilter;
        };

        struct node_t {
            simple_split_t _split;
            qvar_t _q;
            qvar_t _old;
            size_t _parent;
            std::vector<interesect_t> _samples;
            std::unique_ptr<data_t[] > _data = nullptr;
            node_t() = default;
            node_t(const node_t& other, size_t dimen);
            node_t(node_t&& other) noexcept = default;
            node_t& operator=(node_t&& other) noexcept = default;

            size_t find_node(const std::vector<node_t>& nodes, const double * point, const size_t id) const;
            void update(size_t id, bool minimize, const std::vector<MLearning>& clouds, std::vector<node_t>& nodes, size_t dimen, bool allowSplit, const double delta, const propts_t& options);
            std::pair<qvar_t, qvar_t> aggregate_samples(const std::vector<MLearning>& clouds, size_t dimen, bool minimize, std::pair<qvar_t, qvar_t>* tmpq, double discount);
            void print(std::ostream& s, size_t tabs, const std::vector<node_t>& nodes) const;
            void tighten_samples(const std::vector<MLearning>& clouds, size_t cloud);
            void add_sample(size_t dest, const double* f_var, const double* point, double value, size_t dimen, const std::vector<MLearning>& clouds);
            static void update_parents(std::vector<node_t>& nodes, size_t next, bool minimize);
        };

        size_t _dimen = 0;
        std::vector<el_t> _mapping;
        std::vector<node_t> _nodes;
    };
}
#endif /* MLEARNING_H */


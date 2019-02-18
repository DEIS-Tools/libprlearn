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
 * File:   SimpleMLearning.h
 * Author: Peter G. Jensen
 *
 * Created on July 30, 2017, 10:02 AM
 */

#ifndef SIMPLEMLEARNING_H
#define SIMPLEMLEARNING_H

#include "propts.h"
#include "structs.h"
#include <map>

namespace prlearn {

    class SimpleMLearning {
    public:
        SimpleMLearning() = default;
        SimpleMLearning(const SimpleMLearning& other) = default;
        SimpleMLearning& operator=(const SimpleMLearning& other) = default;

        SimpleMLearning(SimpleMLearning&&) = default;
        SimpleMLearning& operator=(SimpleMLearning&&) = default;

        void addSample(size_t dimen, // dimensions
                const double*, const double*, // doubles
                size_t label,
                size_t dest, double value, // cost
                const std::vector<SimpleMLearning>& clouds, // other points
                bool minimization, const double delta,
                const propts_t& options
                );

        void update(const std::vector<SimpleMLearning>& clouds, bool minimization);

        std::pair<double, double> lookup(size_t label, const double*, size_t) const;

        void print(std::ostream& s, size_t tabs, std::map<size_t, size_t>& label_map, const std::vector<SimpleMLearning>&) const;

    protected:

        struct succs_t {
            size_t _nid = 0;
            avg_t _cost;
            bool operator<(const succs_t& other) const;
        };

        struct node_t {
            avg_t _q;
            size_t _label = 0;
            std::vector<succs_t> _succssors;
            bool operator<(const node_t& other) const;
        };
        std::vector<node_t> _nodes;
        avg_t _q;
    };
}
#endif /* SIMPLEMLEARNING_H */


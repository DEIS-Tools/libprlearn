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
 * File:   QLearning.h
 * Author: Peter G. Jensen
 *
 * Created on July 18, 2017, 10:34 AM
 */

#ifndef QLEARNING_H
#define QLEARNING_H
#include "structs.h"

#include <vector>
#include <utility>
#include <numeric>
#include <cmath>
#include <limits>
namespace prlearn {

    template<typename Regressor>
    class QLearning {
    public:
        QLearning() = default;
        QLearning(const QLearning& other) = default;
        QLearning& operator=(const QLearning& other) = default;

        QLearning(QLearning&&) = default;
        QLearning& operator=(QLearning&&) = default;


        void addSample(size_t dimen, // dimensions
                const double* f_val, const double* t_val, // source, destination-states
                size_t label, // action chosen,
                size_t dest, double value, // destination, cost
                const std::vector<QLearning>& clouds, // other points
                bool minimization, const double delta, const propts_t& options);

        void print(std::ostream& s, size_t tabs, std::map<size_t, size_t>& label_map, const std::vector<QLearning<Regressor>>&) const {
            _regressor.print(s, tabs, label_map);
        }

        qvar_t lookup(size_t label, const double* f_var, size_t dimen) const {
            auto res = _regressor.lookup(label, f_var, dimen);
            return res;
        }

    protected:
        Regressor _regressor;
    };

    template<typename Regressor>
    void QLearning<Regressor>::addSample(size_t dimen, // dimensions
            const double* f_var, const double* t_var, // doubles
            size_t label, size_t dest, double value, // cost
            const std::vector<QLearning<Regressor>>&clouds, // other points
            bool minimization, const double delta, const propts_t& options) {
        // The ALPHA part of Q-learning is handled inside the regressors
        auto toDone = 0.0;

        if (dest != 0 && options._discount != 0)
            toDone = clouds[dest]._regressor.getBestQ(t_var, minimization); // 0 is a special sink-node.
        auto nval = value;
        // if future is not a weird number, then add it (discounted)
        if (!std::isinf(toDone) && !std::isnan(toDone)) {
            nval = value + (options._discount * toDone);
        }
        _regressor.update(label, f_var, dimen, nval, delta, options);
    }
}

#endif /* QLEARNING_H */


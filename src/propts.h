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
 * File:   propts.h
 * Author: Peter G. Jensen
 *
 * Created on February 14, 2019, 4:11 PM
 */

#ifndef PROPTS_H
#define PROPTS_H

#include <cstddef>

struct propts_t {
    size_t _q_learn_rate = 10;
    double _upper_t = 1.7;
    double _lower_t = 0.6;
    double _ks_limit = 0.05;
    double _filter_rate = 0.05;
    double _filter_val = 0.05;
    double _discount = 0.99;
    double _indefference = 0.25;
};


#endif /* PROPTS_H */


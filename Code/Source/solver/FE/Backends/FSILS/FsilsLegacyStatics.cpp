/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Array.h"
#include "Array3.h"
#include "Vector.h"

// These legacy container templates (Vector/Array/Array3) are used by the vendored
// FSILS implementation. Their static data members are defined in the legacy
// solver sources (Array.cpp/Vector.cpp/Array3.cpp). In the FE library we provide
// the minimal definitions needed for linking without pulling in the full legacy
// solver infrastructure.

template<>
bool Vector<double>::show_index_check_message = true;
template<>
double Vector<double>::memory_in_use = 0;
template<>
double Vector<double>::memory_returned = 0;
template<>
int Vector<double>::num_allocated = 0;
template<>
int Vector<double>::active = 0;
template<>
bool Vector<double>::write_enabled = false;

template<>
bool Vector<int>::show_index_check_message = true;
template<>
double Vector<int>::memory_in_use = 0;
template<>
double Vector<int>::memory_returned = 0;
template<>
int Vector<int>::num_allocated = 0;
template<>
int Vector<int>::active = 0;
template<>
bool Vector<int>::write_enabled = false;

template<>
bool Vector<Vector<double>>::show_index_check_message = true;
template<>
double Vector<Vector<double>>::memory_in_use = 0;
template<>
double Vector<Vector<double>>::memory_returned = 0;
template<>
int Vector<Vector<double>>::num_allocated = 0;
template<>
int Vector<Vector<double>>::active = 0;
template<>
bool Vector<Vector<double>>::write_enabled = false;

template<>
bool Vector<float>::show_index_check_message = true;
template<>
double Vector<float>::memory_in_use = 0;
template<>
double Vector<float>::memory_returned = 0;
template<>
int Vector<float>::num_allocated = 0;
template<>
int Vector<float>::active = 0;
template<>
bool Vector<float>::write_enabled = false;

template<>
bool Array<bool>::show_index_check_message = true;
template<>
int Array<bool>::id = 0;
template<>
double Array<bool>::memory_in_use = 0;
template<>
double Array<bool>::memory_returned = 0;
template<>
int Array<bool>::num_allocated = 0;
template<>
int Array<bool>::active = 0;
template<>
bool Array<bool>::write_enabled = false;

template<>
bool Array<double>::show_index_check_message = true;
template<>
int Array<double>::id = 0;
template<>
double Array<double>::memory_in_use = 0;
template<>
double Array<double>::memory_returned = 0;
template<>
int Array<double>::num_allocated = 0;
template<>
int Array<double>::active = 0;
template<>
bool Array<double>::write_enabled = false;

template<>
bool Array<int>::show_index_check_message = true;
template<>
int Array<int>::id = 0;
template<>
double Array<int>::memory_in_use = 0;
template<>
double Array<int>::memory_returned = 0;
template<>
int Array<int>::num_allocated = 0;
template<>
int Array<int>::active = 0;
template<>
bool Array<int>::write_enabled = false;

template<>
bool Array3<double>::show_index_check_message = true;
template<>
double Array3<double>::memory_in_use = 0;
template<>
double Array3<double>::memory_returned = 0;
template<>
int Array3<double>::num_allocated = 0;
template<>
int Array3<double>::active = 0;
template<>
bool Array3<double>::write_enabled = false;

template<>
bool Array3<int>::show_index_check_message = true;
template<>
double Array3<int>::memory_in_use = 0;
template<>
double Array3<int>::memory_returned = 0;
template<>
int Array3<int>::num_allocated = 0;
template<>
int Array3<int>::active = 0;
template<>
bool Array3<int>::write_enabled = false;


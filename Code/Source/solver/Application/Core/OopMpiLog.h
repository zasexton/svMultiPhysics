#pragma once

#include "Mesh/Core/MeshComm.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <streambuf>
#include <string>

namespace application {
namespace core {

namespace detail {

inline std::string trim_copy(std::string s)
{
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

inline std::string lower_copy(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

inline bool parse_bool_relaxed(const std::string& raw)
{
  const auto v = lower_copy(trim_copy(raw));
  if (v == "true" || v == "1" || v == "yes" || v == "on") {
    return true;
  }
  if (v == "false" || v == "0" || v == "no" || v == "off") {
    return false;
  }
  return false;
}

class NullBuffer final : public std::streambuf {
public:
  int overflow(int ch) override { return traits_type::not_eof(ch); }
};

inline std::ostream& null_stream()
{
  static NullBuffer buf;
  static std::ostream os(&buf);
  return os;
}

} // namespace detail

inline bool oopTraceEnabled()
{
  if (const char* env = std::getenv("SVMP_OOP_SOLVER_TRACE")) {
    return detail::parse_bool_relaxed(env);
  }
  return false;
}

inline bool oopIsRoot()
{
  return svmp::MeshComm::world().rank() == 0;
}

inline bool oopShouldLog()
{
  return oopIsRoot() || oopTraceEnabled();
}

inline std::ostream& oopCout()
{
  return oopShouldLog() ? std::cout : detail::null_stream();
}

} // namespace core
} // namespace application


#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_DBG_PTR_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_DBG_PTR_H

#include <atomic>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "tensorflow/core/platform/default/stacktrace.h"

namespace aluminum_shark {

// A smart pointer class that behaves identical to std::shared_ptr<T> except
// that it keeps track of where in th code the smart pointers have been created.
// It comes with runtime penalties and is intended for debugging only
template <class T>
class dbg_ptr {
 public:
  explicit dbg_ptr(T* raw = nullptr) {
    id = dbg_ptr::global_count++;
    ptr = std::shared_ptr<T>(raw);
    add(this);
    callers = get_caller();
  };

  ~dbg_ptr() { remove(this); };

  dbg_ptr(const dbg_ptr<T>& r) {
    id = dbg_ptr::global_count++;
    ptr = r.ptr;
    add(this);
    callers = get_caller();
  };

  dbg_ptr& operator=(const dbg_ptr<T>& r) {
    id = dbg_ptr::global_count++;
    ptr = r.ptr;
    add(this);
    callers = get_caller();

    return *this;
  };

  dbg_ptr& operator=(const std::shared_ptr<T>& r) {
    id = dbg_ptr::global_count++;
    ptr = r;
    add(this);
    callers = get_caller();

    return *this;
  };

  dbg_ptr(const std::shared_ptr<T>& r) {
    id = dbg_ptr::global_count++;
    ptr = r;
    add(this);
    callers = get_caller();
  };

  T* get() const { return ptr.get(); };
  T& operator*() const { return ptr.operator*(); };
  T* operator->() const { return ptr.operator->(); };

  long use_count() const { return ptr.use_count(); }

  void reset() { ptr.reset(); }

  explicit operator bool() const { return static_cast<bool>(ptr); }

  operator std::shared_ptr<T>() { return ptr; };

  // operations to inspect the dbg_ptr
  std::vector<std::string> get_caller() {
    std::string bt = tensorflow::CurrentStackTrace();
    size_t start, end = 0;
    // std::vector<std::string> ret;
    // find the firs tab. first caller is dbg_ptr constroctur, which we don't
    // want
    int tab_count = 0;
    /// start = bt.find('\t');
    // we are interestd in the next two lines
    // for (size_t i = 0; i < 2; i++) {
    //   start = bt.find('\t', start + 1);
    //   end = bt.find('\t', start + 1);
    //   start = end;
    //   ret.push_back(bt.substr(start, end - 2));
    // }

    // simpler version. jsut truncate the stacktrace
    for (const auto& c : bt) {
      if (c == '\n') {
        ++tab_count;
        if (tab_count == 6) break;
      }
      ++end;
    }
    return {bt.substr(start, end)};
  };

  // get all dbg_ptr that hold the same object. not thread safe
  const std::vector<dbg_ptr<T>*>& get_all() {
    if (auto search = map.find(ptr.get()); search != map.end()) {
      return search->second;
    }
    return {};
  };

  std::shared_ptr<T>& get_std_shared_ptr() { return ptr; };
  const std::shared_ptr<T>& get_std_shared_ptr() const { return ptr; };

  static std::map<void*, std::vector<dbg_ptr<T>*>>& get_map() { return map; };

  std::vector<std::string> callers;

  static std::atomic_uint64_t global_count;

  uint64_t id;

 private:
  std::shared_ptr<T> ptr;
  std::mutex map_mutex;
  static std::map<void*, std::vector<dbg_ptr<T>*>> map;

  // mangement operations
  void add(dbg_ptr<T>* ptr) {
    const std::lock_guard<std::mutex> lock(map_mutex);
    if (auto search = map.find(ptr->get()); search != map.end()) {
      search->second.push_back(ptr);
    } else {
      map[ptr->get()] = std::vector<dbg_ptr<T>*>{ptr};
    }
  };

  void remove(dbg_ptr<T>* ptr) {
    const std::lock_guard<std::mutex> lock(map_mutex);
    if (auto search = map.find(ptr->get()); search != map.end()) {
      for (size_t i = 0; i < search->second.size();) {
        if (search->second[i]->id == ptr->id) {
          search->second.erase(search->second.begin() + i);
        } else {
          ++i;
        }
      }
    }
  };
};

template <typename T>
std::map<void*, std::vector<aluminum_shark::dbg_ptr<T>*>>
    aluminum_shark::dbg_ptr<T>::map;

template <typename T>
std::atomic_uint64_t aluminum_shark::dbg_ptr<T>::global_count;

template <typename T>
std::ostream& operator<<(std::ostream& os,
                         const aluminum_shark::dbg_ptr<T>& ptr) {
  os << ptr.get_std_shared_ptr();
  return os;
};

template <class T>
bool operator==(const aluminum_shark::dbg_ptr<T>& lhs, std::nullptr_t) {
  return lhs.get_std_shared_ptr() == nullptr;
};

template <class T>
bool operator!=(const aluminum_shark::dbg_ptr<T>& lhs, std::nullptr_t) {
  return lhs.get_std_shared_ptr() != nullptr;
};

// while the replacement is mostly not noticeable, sometimes we need to
// expiliclity convert
template <class T>
const std::shared_ptr<T>& to_std_shared_ptr(const dbg_ptr<T>& ptr) {
  return ptr.get_std_shared_ptr();
};

template <class T>
const std::shared_ptr<T>& to_std_shared_ptr(const std::shared_ptr<T>& ptr) {
  return ptr;
};

template <class T>
std::shared_ptr<T>& to_std_shared_ptr(dbg_ptr<T>& ptr) {
  return ptr.get_std_shared_ptr();
};

template <class T>
std::shared_ptr<T>& to_std_shared_ptr(std::shared_ptr<T>& ptr) {
  return ptr;
};

}  // namespace aluminum_shark

// #define ALUMINUM_SHARK_DEBUG_POINTER

#ifdef ALUMINUM_SHARK_DEBUG_POINTER
template <typename T>
using shared_ptr = ::aluminum_shark::dbg_ptr<T>;
#else
using std::shared_ptr;
#endif

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_DBG_PTR_H \
        */

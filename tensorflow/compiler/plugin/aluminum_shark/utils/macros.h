#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_UTILS_MACROS_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_UTILS_MACROS_H

#if defined(__clang__) || defined(__GNUC__)
#define LIKELY_FALSE(x) __builtin_expect(x, 0)
#define LIKELY_TRUE(x) __builtin_expect(x, 1)
#else
#define LIKELY_FALSE(x) x
#define LIKELY_TRUE(x) x
#endif

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_UTILS_MACROS_H \
        */

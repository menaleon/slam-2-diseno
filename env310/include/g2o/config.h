#ifndef G2O_CONFIG_H
#define G2O_CONFIG_H

/* #undef G2O_HAVE_OPENGL */
#define G2O_OPENGL_FOUND 1
/* #undef G2O_OPENMP */
/* #undef G2O_SHARED_LIBS */
/* #undef G2O_LGPL_SHARED_LIBS */

// available sparse matrix libraries
/* #undef G2O_HAVE_CHOLMOD */
/* #undef G2O_HAVE_CSPARSE */

// logging framework available
/* #undef G2O_HAVE_LOGGING */

#define G2O_CXX_COMPILER "GNU /opt/rh/devtoolset-10/root/usr/bin/c++"
#define G2O_SRC_DIR "/project/g2o"

#endif

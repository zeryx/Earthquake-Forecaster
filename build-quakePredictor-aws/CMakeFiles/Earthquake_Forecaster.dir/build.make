# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/james/Documents/TopCoder/Solutions/git/earthquake_forcaster

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/james/Documents/TopCoder/Solutions/git/earthquake_forcaster/build-quakePredictor-aws

# Include any dependencies generated for this target.
include CMakeFiles/Earthquake_Forecaster.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Earthquake_Forecaster.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Earthquake_Forecaster.dir/flags.make

CMakeFiles/Earthquake_Forecaster.dir/main.cpp.o: CMakeFiles/Earthquake_Forecaster.dir/flags.make
CMakeFiles/Earthquake_Forecaster.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/james/Documents/TopCoder/Solutions/git/earthquake_forcaster/build-quakePredictor-aws/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Earthquake_Forecaster.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Earthquake_Forecaster.dir/main.cpp.o -c /home/james/Documents/TopCoder/Solutions/git/earthquake_forcaster/main.cpp

CMakeFiles/Earthquake_Forecaster.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Earthquake_Forecaster.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/james/Documents/TopCoder/Solutions/git/earthquake_forcaster/main.cpp > CMakeFiles/Earthquake_Forecaster.dir/main.cpp.i

CMakeFiles/Earthquake_Forecaster.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Earthquake_Forecaster.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/james/Documents/TopCoder/Solutions/git/earthquake_forcaster/main.cpp -o CMakeFiles/Earthquake_Forecaster.dir/main.cpp.s

CMakeFiles/Earthquake_Forecaster.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/Earthquake_Forecaster.dir/main.cpp.o.requires

CMakeFiles/Earthquake_Forecaster.dir/main.cpp.o.provides: CMakeFiles/Earthquake_Forecaster.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/Earthquake_Forecaster.dir/build.make CMakeFiles/Earthquake_Forecaster.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/Earthquake_Forecaster.dir/main.cpp.o.provides

CMakeFiles/Earthquake_Forecaster.dir/main.cpp.o.provides.build: CMakeFiles/Earthquake_Forecaster.dir/main.cpp.o

# Object files for target Earthquake_Forecaster
Earthquake_Forecaster_OBJECTS = \
"CMakeFiles/Earthquake_Forecaster.dir/main.cpp.o"

# External object files for target Earthquake_Forecaster
Earthquake_Forecaster_EXTERNAL_OBJECTS =

Earthquake_Forecaster: CMakeFiles/Earthquake_Forecaster.dir/main.cpp.o
Earthquake_Forecaster: CMakeFiles/Earthquake_Forecaster.dir/build.make
Earthquake_Forecaster: /usr/local/cuda/lib64/libcudart.so
Earthquake_Forecaster: dlib_build/libdlib.a
Earthquake_Forecaster: /usr/lib/x86_64-linux-gnu/libpthread.so
Earthquake_Forecaster: /usr/lib/x86_64-linux-gnu/libnsl.so
Earthquake_Forecaster: /usr/lib/x86_64-linux-gnu/libSM.so
Earthquake_Forecaster: /usr/lib/x86_64-linux-gnu/libICE.so
Earthquake_Forecaster: /usr/lib/x86_64-linux-gnu/libX11.so
Earthquake_Forecaster: /usr/lib/x86_64-linux-gnu/libXext.so
Earthquake_Forecaster: /usr/lib/x86_64-linux-gnu/libpng.so
Earthquake_Forecaster: /usr/lib/x86_64-linux-gnu/libjpeg.so
Earthquake_Forecaster: /usr/lib/libcblas.so
Earthquake_Forecaster: /usr/lib/liblapack.so
Earthquake_Forecaster: /usr/lib/x86_64-linux-gnu/libsqlite3.so
Earthquake_Forecaster: CMakeFiles/Earthquake_Forecaster.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable Earthquake_Forecaster"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Earthquake_Forecaster.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Earthquake_Forecaster.dir/build: Earthquake_Forecaster
.PHONY : CMakeFiles/Earthquake_Forecaster.dir/build

CMakeFiles/Earthquake_Forecaster.dir/requires: CMakeFiles/Earthquake_Forecaster.dir/main.cpp.o.requires
.PHONY : CMakeFiles/Earthquake_Forecaster.dir/requires

CMakeFiles/Earthquake_Forecaster.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Earthquake_Forecaster.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Earthquake_Forecaster.dir/clean

CMakeFiles/Earthquake_Forecaster.dir/depend:
	cd /home/james/Documents/TopCoder/Solutions/git/earthquake_forcaster/build-quakePredictor-aws && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/james/Documents/TopCoder/Solutions/git/earthquake_forcaster /home/james/Documents/TopCoder/Solutions/git/earthquake_forcaster /home/james/Documents/TopCoder/Solutions/git/earthquake_forcaster/build-quakePredictor-aws /home/james/Documents/TopCoder/Solutions/git/earthquake_forcaster/build-quakePredictor-aws /home/james/Documents/TopCoder/Solutions/git/earthquake_forcaster/build-quakePredictor-aws/CMakeFiles/Earthquake_Forecaster.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Earthquake_Forecaster.dir/depend


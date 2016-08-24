# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/parallels/src/kms-opencv-plugin-sample

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu

# Include any dependencies generated for this target.
include src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/depend.make

# Include the progress variables for this target.
include src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/progress.make

# Include the compile flags for this target's objects.
include src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/flags.make

src/server/cpp_server_internal.generated: ../src/server/interface/opencvpluginsample.kmd.json
src/server/cpp_server_internal.generated: ../src/server/interface/opencvpluginsample.OpencvPluginSample.kmd.json
	$(CMAKE_COMMAND) -E cmake_progress_report /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Generating cpp_server_internal.generated, implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp, implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp, implementation/generated-cpp/OpencvPluginSampleImplFactory.hpp"
	cd /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server && /usr/bin/cmake -E touch cpp_server_internal.generated
	cd /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server && /usr/bin/kurento-module-creator -c /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server/implementation/generated-cpp -r /home/parallels/src/kms-opencv-plugin-sample/src/server/interface -dr /usr/share/kurento/modules -it cpp_server_internal

src/server/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp: src/server/cpp_server_internal.generated

src/server/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp: src/server/cpp_server_internal.generated

src/server/implementation/generated-cpp/OpencvPluginSampleImplFactory.hpp: src/server/cpp_server_internal.generated

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.o: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/flags.make
src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.o: ../src/server/implementation/objects/OpencvPluginSampleImpl.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.o"
	cd /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server && /usr/bin/x86_64-linux-gnu-g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.o -c /home/parallels/src/kms-opencv-plugin-sample/src/server/implementation/objects/OpencvPluginSampleImpl.cpp

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.i"
	cd /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server && /usr/bin/x86_64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/parallels/src/kms-opencv-plugin-sample/src/server/implementation/objects/OpencvPluginSampleImpl.cpp > CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.i

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.s"
	cd /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server && /usr/bin/x86_64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/parallels/src/kms-opencv-plugin-sample/src/server/implementation/objects/OpencvPluginSampleImpl.cpp -o CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.s

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.o.requires:
.PHONY : src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.o.requires

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.o.provides: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.o.requires
	$(MAKE) -f src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/build.make src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.o.provides.build
.PHONY : src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.o.provides

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.o.provides.build: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.o

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.o: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/flags.make
src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.o: ../src/server/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.o"
	cd /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server && /usr/bin/x86_64-linux-gnu-g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.o -c /home/parallels/src/kms-opencv-plugin-sample/src/server/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.i"
	cd /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server && /usr/bin/x86_64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/parallels/src/kms-opencv-plugin-sample/src/server/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp > CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.i

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.s"
	cd /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server && /usr/bin/x86_64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/parallels/src/kms-opencv-plugin-sample/src/server/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp -o CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.s

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.o.requires:
.PHONY : src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.o.requires

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.o.provides: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.o.requires
	$(MAKE) -f src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/build.make src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.o.provides.build
.PHONY : src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.o.provides

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.o.provides.build: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.o

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.o: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/flags.make
src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.o: src/server/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.o"
	cd /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server && /usr/bin/x86_64-linux-gnu-g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.o -c /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.i"
	cd /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server && /usr/bin/x86_64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp > CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.i

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.s"
	cd /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server && /usr/bin/x86_64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp -o CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.s

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.o.requires:
.PHONY : src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.o.requires

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.o.provides: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.o.requires
	$(MAKE) -f src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/build.make src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.o.provides.build
.PHONY : src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.o.provides

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.o.provides.build: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.o

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.o: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/flags.make
src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.o: src/server/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.o"
	cd /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server && /usr/bin/x86_64-linux-gnu-g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.o -c /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.i"
	cd /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server && /usr/bin/x86_64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp > CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.i

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.s"
	cd /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server && /usr/bin/x86_64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp -o CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.s

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.o.requires:
.PHONY : src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.o.requires

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.o.provides: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.o.requires
	$(MAKE) -f src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/build.make src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.o.provides.build
.PHONY : src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.o.provides

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.o.provides.build: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.o

# Object files for target kmsopencvpluginsampleimpl
kmsopencvpluginsampleimpl_OBJECTS = \
"CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.o" \
"CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.o" \
"CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.o" \
"CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.o"

# External object files for target kmsopencvpluginsampleimpl
kmsopencvpluginsampleimpl_EXTERNAL_OBJECTS =

src/server/libkmsopencvpluginsampleimpl.so.0.0.1~12.gd26e75d: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.o
src/server/libkmsopencvpluginsampleimpl.so.0.0.1~12.gd26e75d: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.o
src/server/libkmsopencvpluginsampleimpl.so.0.0.1~12.gd26e75d: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.o
src/server/libkmsopencvpluginsampleimpl.so.0.0.1~12.gd26e75d: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.o
src/server/libkmsopencvpluginsampleimpl.so.0.0.1~12.gd26e75d: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/build.make
src/server/libkmsopencvpluginsampleimpl.so.0.0.1~12.gd26e75d: /usr/lib/x86_64-linux-gnu/libkmselementsimpl.so
src/server/libkmsopencvpluginsampleimpl.so.0.0.1~12.gd26e75d: /usr/lib/x86_64-linux-gnu/libkmsfiltersimpl.so
src/server/libkmsopencvpluginsampleimpl.so.0.0.1~12.gd26e75d: /usr/lib/x86_64-linux-gnu/libkmselementsimpl.so
src/server/libkmsopencvpluginsampleimpl.so.0.0.1~12.gd26e75d: src/server/libkmsopencvpluginsampleinterface.a
src/server/libkmsopencvpluginsampleimpl.so.0.0.1~12.gd26e75d: /usr/lib/x86_64-linux-gnu/libkmselementsimpl.so
src/server/libkmsopencvpluginsampleimpl.so.0.0.1~12.gd26e75d: /usr/lib/x86_64-linux-gnu/libkmsfiltersimpl.so
src/server/libkmsopencvpluginsampleimpl.so.0.0.1~12.gd26e75d: /usr/lib/x86_64-linux-gnu/libkmselementsimpl.so
src/server/libkmsopencvpluginsampleimpl.so.0.0.1~12.gd26e75d: /usr/lib/x86_64-linux-gnu/libkmsfiltersimpl.so
src/server/libkmsopencvpluginsampleimpl.so.0.0.1~12.gd26e75d: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library libkmsopencvpluginsampleimpl.so"
	cd /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kmsopencvpluginsampleimpl.dir/link.txt --verbose=$(VERBOSE)
	cd /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server && $(CMAKE_COMMAND) -E cmake_symlink_library "libkmsopencvpluginsampleimpl.so.0.0.1~12.gd26e75d" libkmsopencvpluginsampleimpl.so.0 libkmsopencvpluginsampleimpl.so

src/server/libkmsopencvpluginsampleimpl.so.0: src/server/libkmsopencvpluginsampleimpl.so.0.0.1~12.gd26e75d

src/server/libkmsopencvpluginsampleimpl.so: src/server/libkmsopencvpluginsampleimpl.so.0.0.1~12.gd26e75d

# Rule to build all files generated by this target.
src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/build: src/server/libkmsopencvpluginsampleimpl.so
.PHONY : src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/build

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/requires: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleImpl.cpp.o.requires
src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/requires: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/objects/OpencvPluginSampleOpenCVImpl.cpp.o.requires
src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/requires: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp.o.requires
src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/requires: src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp.o.requires
.PHONY : src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/requires

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/clean:
	cd /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server && $(CMAKE_COMMAND) -P CMakeFiles/kmsopencvpluginsampleimpl.dir/cmake_clean.cmake
.PHONY : src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/clean

src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/depend: src/server/cpp_server_internal.generated
src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/depend: src/server/implementation/generated-cpp/SerializerExpanderOpencvpluginsample.cpp
src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/depend: src/server/implementation/generated-cpp/OpencvPluginSampleImplInternal.cpp
src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/depend: src/server/implementation/generated-cpp/OpencvPluginSampleImplFactory.hpp
	cd /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/parallels/src/kms-opencv-plugin-sample /home/parallels/src/kms-opencv-plugin-sample/src/server /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server /home/parallels/src/kms-opencv-plugin-sample/obj-x86_64-linux-gnu/src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/server/CMakeFiles/kmsopencvpluginsampleimpl.dir/depend


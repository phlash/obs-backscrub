# obs-backscrub

Integration for [backscrub](https://github.com/floe/backscrub) project into OBS Studio

## What is this?

It's a video filter plugin for OBS Studio (currently limited to video capture devices that produce YUY2 format streams), which uses the `backscrub`
library to remove the background of a video source (replacing it with green), allowing subsequent chroma keying of alternate backgrounds.

## Neat! How do I build it?

### Linux

 * Install the dependencies (assuming you have OBS Studio already!): `build-essentials`, `libobs-dev`, `libopencv-dev`
 * Clone this project.
 * Do the CMake dance:
   ```bash
   % mkdir build; cmake -B build; cmake --build build -j
   ```
This will pull the submodules (backscrub and Tensorflow), configure them (mostly Tensorflow fetching several more dependencies) then compile everything.
Expect to wait 10+ minutes for a full build, and lose ~2GB of disk space.

### Windows

__NB: This is fiddly, fragile and poorly tested__

 * Development environment: VS2019 community edition, on Server2019 / Windows10/11
 * Download and install dependencies (assuming you have OBS Studio already!):
   * OpenCV 3.4.14: https://sourceforge.net/projects/opencvlibrary/files/3.4.14/opencv-3.4.14-vc14_vc15.exe/download
   * Note installed path, eg: `C:\Users\phlash\Download\opencv`
 * Download OBS source code and bodge an import library (in lieu of a libobs-dev package):
   * Match your version, eg for 29.0.2: https://github.com/obsproject/obs-studio/archive/refs/tags/29.0.2.zip
   * Create a fresh `libobs` folder eg: `C:\Users\phlash\Downloads\libobs`
   * Save [LibObsConfig.zip](https://github.com/phlash/obs-backscrub/files/11357427/LibObsConfig.zip) to the `libobs` folder and unpack in place to get a CMake config script
   * From OBS source ZIP, unpack only the `libobs` folder _into folder above_, then _rename as `inc`_, eg: `C:\Users\phlash\Downloads\libobs\inc`
   * Create a `bin` folder next to `inc` eg: `C:\Users\phlash\Downloads\libobs\bin`
   * Copy the `OBS.DLL` library out of your installed OBS Studio eg: `C:\Program Files\obs-studio\bin\64bit\OBS.DLL` into `bin` folder.
   * In a `VS2019 Developer Command Prompt`, generate an import library from the DLL, instructions modified from: https://stackoverflow.com/questions/9946322/how-to-generate-an-import-library-lib-file-from-a-dll
     ```cmd
     C> cd <libobs\bin folder>
     C> echo LIBRARY OBS > obs.def
     C> echo EXPORTS >> obs.def
     C> for /f "skip=19 tokens=4" %A in ('dumpbin /exports OBS.DLL') do echo %A >> obs.def
     C> lib /def:obs.def /out:OBS.LIB /machine:x64
     ```
   * Clone this project.
   * In a `VS2019 Developer Command Prompt` Do the CMake dance, informing it where OpenCV and libobs are:
   ```cmd
   C> mkdir build
   C> cmake -B build -D CMAKE_PREFIX_PATH="<OpenCV path>\build;<libobs folder>"
   C> cmake --build build -j
   ```
   * The build will fail, you need to fix up a bug in Tensorflow (https://github.com/tensorflow/tensorflow/issues/54323)
     * Edit: `tensorflow\tensorflow\core\lib\random\random_distributions_utils.h` and replace `M_PI` symbol with `3.1415928`.
   * Re-run build:
   ```cmd
   C> cmake --build build -j
   ```

## It's built - how do I install it?

### Linux

I choose to create sym-links for the built object and the data directory out of the obs-studio installation, this avoids install-to-test issues:
```bash
% cd /lib/x86_64-linux-gnu/obs-plugins
% sudo ln -s /home/phlash/obs-backscrub/build/obs-backscrub.so .
% cd /usr/share/obs/obs-plugins
% sudo ln -s /home/phlash/obs-backscrub/data obs-backscrub
```

### Windows

Not quite the same as Linux as Windows cannot do file symlinks, only folders, so we copy the built object and dependencies:
```cmd
C> cd \Program Files\obs-studio\obs-plugin\64bit
C> copy \Users\phlash\obs-backscrub\build\obs-backscrub.dll
C> copy <OpenCV path>\build\x64\opencv-world3414d.dll
C> cd ..\..\data\obs-plugin
C> mklink /D obs-backscrub \Users\phlash\obs-backscrub\data
```

## Using it?

Fire up OBS Studio - _check the logs_ to ensure `obs-backscrub.dll` loads successfully.

Add a video capture source, go to filters, and add an audio/video filter 'Background scrubber'. That's it.

#!/bin/bash

__BuildOS=""
__CleanBuild=0

case $OSTYPE in
  msys|cygwin)
  __BuildOS=win
  ;;
  *)
  __BuildOS=linux
  ;;
esac

__BuildArch=x64
__BuildType=Debug
__COMPILER=gcc
__CC=gcc
__CXX=g++

for i in "$@"
  do
    lowerI=${i,,}
    case $lowerI in
      -?|-h|--help)
      usage
      exit 1
    ;;
    x64)
      __BuildArch=x64
      ;;
    x86)
      __BuildArch=x86
    ;;        
    debug)
      __BuildType=Debug
      ;;
    release)
      __BuildType=Release
    ;;
    iaca)
      __BuildType=IACA
    ;;
    clang)
      __COMPILER=clang
      __CC=clang-6.0
      __CXX=clang++-6.0
    ;;
    gcc)
      __COMPILER=gcc
      __CC=gcc
      __CXX=g++
    ;;
    test)
      __RunTests=1
    ;;
    clean)
      __CleanBuild=1
    ;;
    *)
      __UnprocessedBuildArgs="$__UnprocessedBuildArgs $i"          
  esac
done

if [ $__CleanBuild == "1" ]; then
  rm -rf dist
  rm -rf build-{debug,release}
  exit 0
fi

__DistDir=build-${__BuildType,,}-${__COMPILER}

mkdir -p ${__DistDir}
pushd ${__DistDir}

if [ $__BuildOS == "win" ]; then
  cmake -G "Visual Studio 15 2017 Win64" -DCMAKE_BUILD_TYPE=${__BuildType^^} ..
  
  vs=$(/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/Installer/vswhere.exe -latest | grep installationPath | cut -f 2- -d : -d " ") 
  __MSBuildExePath="$vs/MSBuild/15.0/Bin/MSBuild.exe"      
  if [ ! -f "$__MSBuildExePath" ]; then
    echo Error: Could not find MSBuild.exe
    exit 1
  fi
  ${__MSBuildExePath} -p:Platform=${__BuildArch} -p:Configuration=${__BuildType} dodo.sln
  build_result=$?
fi

if [ $__BuildOS == "linux" ]; then
  CC=${__CC} CXX=${__CXX} cmake -DCMAKE_BUILD_TYPE=${__BuildType^^} ..
  make -j4
  build_result=$?
fi


build_result=$?

if [ "$__RunTests" == "1" ]; then
  ./tests/bitgoo_tests
fi

# Build complete
if [ ${build_result} == 0 ]; then
    echo bitgoo successfully built. ✔
    echo "binaries are available at ${__DistDir}"
else
    echo "build failed miserably (${build_result}), you suck, 💩💩💩"
    exit $build_result
fi
popd


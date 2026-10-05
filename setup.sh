#!/bin/zsh
# Meld Turbo setup: fetch the llama.cpp MTP branch, apply the Meld Turbo patches and build with Metal.
# Models are downloaded separately (see README).
#
#   ./setup.sh                    # installs into ~/.meld-turbo
#   MELD_HOME=/path ./setup.sh
set -e
here=$(cd $(dirname $0) && pwd)
home=${MELD_HOME:-$HOME/.meld-turbo}
src=$home/llama.cpp
base=6fcaa16f4b360649933a54d1f91ad40ed35c0e11   # danielhanchen/llama.cpp, branch qwen4exp/mtp

mkdir -p $home
if [ ! -d $src/.git ]; then
  git init -q $src
  git -C $src remote add origin https://github.com/danielhanchen/llama.cpp.git
fi
git -C $src fetch -q --depth 1 origin $base
git -C $src checkout -q -B meld-turbo $base
git -C $src -c user.name=meld-turbo -c user.email=meld-turbo@localhost am -q --3way $here/patches/*.patch
cmake -S $src -B $src/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > /dev/null
cmake --build $src/build -j $(sysctl -n hw.ncpu) --target llama-server llama-bench

# MTP draft vocabulary subset (106,299 common token ids)
cp $here/data/draft_vocab.bin $home/draft_vocab.bin

echo "built: $src/build/bin/llama-server"
echo "next: download the models into $home/models (README), then ./serve.sh"

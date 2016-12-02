#!/usr/bin/env bash

set -e
set -o nounset

basedir=$(cd $(dirname $0)/..;pwd)

function fat_echo() {
    echo "############################################"
    echo "########## $1"
}

function wget_or_curl() {
  [ $# -eq 2 ] || { echo "Usage: wget_or_curl <url> <fpath>" && exit 1; }
  if type wget &> /dev/null; then
    local download_cmd="wget -T 10 -t 3 -O"
  else
    local download_cmd="curl -L -o"
  fi
  $download_cmd "$2" "$1"
}

target="icwb2-data"
if [ ! -e "$basedir/data/$target" ]; then
    fat_echo "Downloading datasets from www.sighan.org"
    (
        # wget_or_curl http://www.sighan.org/bakeoff2005/data/$target.zip $basedir/data/$target.zip  This site now is not available.
        wget_or_curl https://github.com/chantera/$target/archive/master.zip $basedir/data/$target.zip
        unzip -q -d $basedir/data/ $basedir/data/$target.zip
        rm $basedir/data/$target.zip
        mv $basedir/data/$target-master $basedir/data/$target
    )
fi

# you can download the latest articles manually.
# https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
target="zhwiki-20161001-pages-articles.xml.bz2"
if [ ! -e "$basedir/data/$target" ]; then
    fat_echo "Downloading articles from dumps.wikimedia.org"
    (
        wget_or_curl https://dumps.wikimedia.org/zhwiki/20161001/$target $basedir/data/$target
        # bunzip2 $basedir/data/$target.bz2
    )
fi

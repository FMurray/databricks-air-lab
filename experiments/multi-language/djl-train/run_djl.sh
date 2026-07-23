#!/bin/bash
# Polyglot ladder step 1: portable JRE + DJL fat jar, no Docker.
# Assumes probe.sh said /tmp is exec-ok. JRE source order:
#   1. $JRE_TARBALL (stage in a UC volume for the no-egress path)
#   2. Adoptium download (probe.sh verifies this egress)
set -euo pipefail

WORK=/tmp/polyglot
mkdir -p "$WORK"
JAR="$CODE_SOURCE_PATH/experiments/multi-language/djl-train/build/libs/djl-train-all.jar"
[ -f "$JAR" ] || { echo "fat jar missing — build locally with 'gradle shadowJar' first"; exit 1; }

if [ -n "${JRE_TARBALL:-}" ] && [ -f "$JRE_TARBALL" ]; then
  tar -xzf "$JRE_TARBALL" -C "$WORK"
else
  echo "downloading Temurin 17 JRE (linux x64) from Adoptium..."
  curl -fsSL --max-time 300 -o "$WORK/jre.tar.gz" \
    "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jre/hotspot/normal/eclipse"
  tar -xzf "$WORK/jre.tar.gz" -C "$WORK"
fi

export JAVA_HOME
JAVA_HOME="$(dirname "$(dirname "$(find "$WORK" -name java -type f | head -1)")")"
echo "PROBE:java_home=$JAVA_HOME"
"$JAVA_HOME/bin/java" -version

# DJL caches (and, unless vendored, downloads) native libtorch here at first run
export DJL_CACHE_DIR=/tmp/djl-cache

exec "$JAVA_HOME/bin/java" -jar "$JAR" "${EPOCHS:-2}"

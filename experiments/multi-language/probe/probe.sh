#!/bin/bash
# Exec-environment probe for the polyglot-training ladder (step 0).
# Answers: can we run uploaded binaries on an AIR node, and what native ABI do we target?
# Every check prints PROBE:<key>=<value> so results grep cleanly out of run logs.

echo "PROBE:uname=$(uname -a)"
echo "PROBE:os=$(head -1 /etc/os-release 2>/dev/null)"
echo "PROBE:glibc=$(ldd --version 2>/dev/null | head -1)"
echo "PROBE:nproc=$(nproc)"
echo "PROBE:java=$(command -v java || echo MISSING)"
echo "PROBE:gcc=$(command -v gcc || echo MISSING)"
echo "PROBE:code_source_path=$CODE_SOURCE_PATH"

probe_dir="$CODE_SOURCE_PATH/experiments/multi-language/probe"

# Mount flags — the noexec question
echo "PROBE:snapshot_mount=$(findmnt -no OPTIONS -T "$probe_dir" 2>/dev/null || echo unknown)"
echo "PROBE:tmp_mount=$(findmnt -no OPTIONS -T /tmp 2>/dev/null || echo unknown)"

# Does the snapshot preserve the +x bit set in git?
ls -l "$probe_dir/exec_test.sh"
if "$probe_dir/exec_test.sh" 2>/dev/null; then
  echo "PROBE:snapshot_script_exec=ok"
else
  chmod +x "$probe_dir/exec_test.sh" 2>/dev/null \
    && "$probe_dir/exec_test.sh" 2>/dev/null \
    && echo "PROBE:snapshot_script_exec=ok_after_chmod" \
    || echo "PROBE:snapshot_script_exec=blocked"
fi

# Can an ELF binary exec from the snapshot path? (copy a known-good system binary in)
if cp /bin/true "$probe_dir/true_copy" 2>/dev/null && "$probe_dir/true_copy" 2>/dev/null; then
  echo "PROBE:snapshot_binary_exec=ok"
else
  echo "PROBE:snapshot_binary_exec=blocked_or_readonly"
fi
cp /bin/true /tmp/true_copy && /tmp/true_copy && echo "PROBE:tmp_binary_exec=ok" \
  || echo "PROBE:tmp_binary_exec=blocked"

# Egress: Maven Central + Adoptium (JRE download path for ladder step 1)
for url in https://repo1.maven.org/maven2/ https://api.adoptium.net/v3/info/available_releases; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$url" || echo FAIL)
  echo "PROBE:egress:$url=$code"
done

# GPU visibility baseline (compare later with what the JVM sees)
command -v nvidia-smi >/dev/null && nvidia-smi -L | sed 's/^/PROBE:gpu=/'
echo "PROBE:done=true"

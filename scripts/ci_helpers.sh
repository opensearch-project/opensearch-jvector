#!/bin/sh

# Sources:
# https://github.com/datastax/jvector/blob/4f13ff42e93f2e7b16020de3aaeefe9372e0dd36/jvector-native/src/main/native/tests/test_x86_cpu_features.cpp#L43-L89
# https://github.com/datastax/jvector/blob/4f13ff42e93f2e7b16020de3aaeefe9372e0dd36/jvector-native/src/test/java/io/github/jbellis/jvector/vector/DispatcherCpuFlagsTest.java#L239-L271
#
# Last grep uses -q

get_avx_support() {
  if lscpu | grep -i avx512f | grep -i avx512bw | grep -i avx512cd | grep -i avx512dq | grep -qi avx512vl; then
    if lscpu | grep -i avx512_vnni | grep -i avx512vbmi | grep -i avx512_vbmi2 | grep -i avx512ifma | grep -i avx512_bitalg | grep -i avx512avx512_vpopcntdq | grep -i gfni | grep -i vaes | grep -qi vpclmulqdq; then
      if lscpu | grep -qi avx512_fp16; then
        echo "AVX3_SPR"
      else
        echo "AVX3_DL"
      fi
    else
        echo "AVX3"
    fi
  elif lscpu | grep -qi avx2; then
    echo "AVX2"
  else
    echo "NO_AVX"
  fi
}

print_avx_support_message() {
  avx_support=$1
  if [ "$avx_support" = "AVX3_SPR" ]; then
    echo "avx512_spr available on system (Intel(R) Sapphire Rapids or newer)"
  elif [ "$avx_support" = "AVX3_DL" ]; then
    echo "avx512_dl available on system (Intel(R) Ice Lake or newer)"
  elif [ "$avx_support" = "AVX3" ]; then
    echo "avx512 available on system"
  elif [ "$avx_support" = "AVX2" ]; then
    echo "avx2 available on system"
  else
    echo "avx512 and avx2 not available on system"
  fi
}

check_avx_support() {
  avx_support=$(get_avx_support)
  print_avx_support_message "$avx_support"
  echo "$avx_support"
}

get_avx_build_flags() {
  avx_support=$1
  if [ "$avx_support" = "AVX3_SPR" ]; then
    echo "-Davx512_spr.enabled=true"
  elif [ "$avx_support" = "AVX3_DL" ]; then
    echo "-Davx512_dl.enabled=true"
  elif [ "$avx_support" = "AVX3" ]; then
    echo "-Davx512.enabled=true"
  elif [ "$avx_support" = "AVX2" ]; then
    echo "-Davx2.enabled=true"
  else
    echo "-Davx2.enabled=false -Davx512.enabled=false -Davx512_dl.enabled=false -Davx512_spr.enabled=false -Dsimd.enabled=false"
  fi
}

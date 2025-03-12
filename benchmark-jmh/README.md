# Benchmarking JVector k-NN Plugin

This directory contains benchmarks for the JVector k-NN Plugin. The benchmarks are written using the JMH framework and are located in the `src/jmh/java` directory. The benchmarks are designed to measure the performance of the JVector k-NN Plugin in various scenarios.

To run the benchmarks, use the following command:

```shell
./gradlew benchmark-jmh:jmh -PjmhInclude=<benchmark_class_name>
```

Replace `<benchmark_class_name>` with the name of the benchmark class you want to run. For example, to run the `FormatBenchmarkRandomVectors` benchmark, use the following command:

```shell
./gradlew benchmark-jmh:jmh -PjmhInclude=org.opensearch.knn.index.codec.jvector.FormatBenchmarkRandomVectors
```
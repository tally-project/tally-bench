def get_benchmark_str(model_name, batch_size, mixed_precision, bench_id=None):
    mp_str = "with" if mixed_precision else "no"
    benchmark_id = f"{model_name}_{batch_size}_{mp_str}_mp"
    if bench_id is not None:
        benchmark_id += f"_{bench_id}"
    return benchmark_id
